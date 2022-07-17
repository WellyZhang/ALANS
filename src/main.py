# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import warnings

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

import criteria
import model
from const import DIM_COLOR, DIM_EXIST, DIM_SIZE, DIM_TYPE, MATRIX_SIZE
from dataset import Dataset

# warnings.filterwarnings("ignore", category=DeprecationWarning)


def check_paths(args):
    timestamp = time.ctime().replace(" ", "-")
    args.save_dir = os.path.join(args.save_dir, timestamp)
    args.log_dir = os.path.join(args.log_dir, timestamp)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, timestamp)
    try:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)
    except OSError as e:
        print(e)
        sys.exit(1)


def get_model(model_name):
    class_name = "".join(model_name.title().split("_"))
    return getattr(model, class_name)


def train(args, device):

    def train_epoch(epoch, steps):
        model.train()
        xe_avg = 0.0
        train_loader_iter = iter(train_loader)
        for _ in trange(len(train_loader_iter)):
            steps += 1
            images, targets, rule_annotations = next(train_loader_iter)
            images = images.to(device)
            targets = targets.to(device)
            for idx in range(len(rule_annotations)):
                rule_annotations[idx] = rule_annotations[idx].to(device)
            spaces = model.update_spaces()
            panel_attr_probs, attr_inner_objs, attr_pred_probs, object_attr_logprob = model.forward(
                images, *spaces)
            attr_action_probs = model.get_action_probs(panel_attr_probs,
                                                       attr_inner_objs)
            exist_loss, number_loss, type_loss, size_loss, color_loss = model.action_auxiliary_loss(
                attr_action_probs, [rule_annotations])
            xe_loss, scores = loss_func(panel_attr_probs, attr_pred_probs,
                                        attr_action_probs, error_func, targets)
            final_loss = xe_loss + args.aux * (
                exist_loss + number_loss + type_loss + size_loss + color_loss)
            acc = criteria.calculate_acc(scores, targets)
            xe_avg += xe_loss.item()
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()
            writer.add_scalar("Train Loss", final_loss.item(), steps)
            writer.add_scalar("Train XE", xe_loss.item(), steps)
            writer.add_scalar("Train Acc", acc, steps)
        print("Epoch {}, Train Avg XE: {:.6f}".format(
            epoch, xe_avg / float(len(train_loader_iter))))

        return steps

    def validate_epoch(epoch, steps):
        model.eval()
        acc_avg = 0.0
        xe_avg = 0.0
        total = 0
        val_loader_iter = iter(val_loader)
        spaces = model.update_spaces()
        for _ in trange(len(val_loader_iter)):
            images, targets, _ = next(val_loader_iter)
            total += images.shape[0]
            images = images.to(device)
            targets = targets.to(device)
            panel_attr_probs, attr_inner_objs, attr_pred_probs, object_attr_logprob = model.forward(
                images, *spaces)
            attr_action_probs = model.get_action_probs(panel_attr_probs,
                                                       attr_inner_objs)
            xe_loss, scores = loss_func(panel_attr_probs, attr_pred_probs,
                                        attr_action_probs, error_func, targets)
            xe_avg += xe_loss.item() * images.shape[0]
            acc = criteria.calculate_correct(scores, targets)
            acc_avg += acc
        writer.add_scalar("Valid Avg XE", xe_avg / float(total), steps)
        writer.add_scalar("Valid Avg Acc", acc_avg / float(total), steps)
        print("Epoch {}, Valid Avg XE: {:.6f}, Valid Avg Acc: {:.4f}".format(
            epoch, xe_avg / float(total), acc_avg / float(total)))
        return xe_avg / float(total)

    def test_epoch(epoch, steps):
        model.eval()
        acc_avg = 0.0
        xe_avg = 0.0
        total = 0
        test_loader_iter = iter(test_loader)
        spaces = model.update_spaces()
        for _ in trange(len(test_loader_iter)):
            images, targets, _ = next(test_loader_iter)
            total += images.shape[0]
            images = images.to(device)
            targets = targets.to(device)
            panel_attr_probs, attr_inner_objs, attr_pred_probs, object_attr_logprob = model.forward(
                images, *spaces)
            attr_action_probs = model.get_action_probs(panel_attr_probs,
                                                       attr_inner_objs)
            xe_loss, scores = loss_func(panel_attr_probs, attr_pred_probs,
                                        attr_action_probs, error_func, targets)
            xe_avg += xe_loss.item() * images.shape[0]
            acc = criteria.calculate_correct(scores, targets)
            acc_avg += acc
        writer.add_scalar("Test Avg XE", xe_avg / float(total), steps)
        writer.add_scalar("Test Avg Acc", acc_avg / float(total), steps)
        print("Epoch {}, Test  Avg XE: {:.6f}, Test  Avg Acc: {:.4f}".format(
            epoch, xe_avg / float(total), acc_avg / float(total)))

    # set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # init model
    model = get_model(args.config)(DIM_EXIST,
                                   DIM_TYPE,
                                   DIM_SIZE,
                                   DIM_COLOR,
                                   MATRIX_SIZE,
                                   solve_mode=args.solve_mode,
                                   execute_mode=args.execute_mode)
    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume))
    # model.mute_module("number_sys")
    # model.mute_module("object_cnn.exist_model")
    # model.mute_module("object_cnn.type_model")
    # model.mute_module("object_cnn.size_model")
    # model.mute_module("object_cnn.color_model")
    if args.cuda and args.multigpu and torch.cuda.device_count() > 1:
        print("Running the model on {} GPUs".format(torch.cuda.device_count()))
        model.parallelize()
    model.to(device)
    optimizer = optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        args.lr,
        weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     "min",
                                                     factor=0.5,
                                                     verbose=True)
    loss_func = getattr(model, "loss_" + args.loss)
    error_func = getattr(criteria, args.error)

    # dataset loader
    train_set = Dataset(args.dataset, "train", args.img_size, args.config)
    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)
    val_set = Dataset(args.dataset,
                      "val",
                      args.img_size,
                      args.config,
                      test=True)
    val_loader = DataLoader(val_set,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers)
    test_set = Dataset(args.dataset,
                       "test",
                       args.img_size,
                       args.config,
                       test=True)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)

    # logging
    writer = SummaryWriter(args.log_dir)
    total_steps = 0

    # training loop starts
    for epoch in range(args.epochs):
        total_steps = train_epoch(epoch, total_steps)
        with torch.no_grad():
            val_loss = validate_epoch(epoch, total_steps)
            test_epoch(epoch, total_steps)
        scheduler.step(val_loss)

        # save checkpoint
        model.eval()
        model.cpu()
        ckpt_model_name = "epoch_{}_batch_{}_seed_{}_config_{}_img_{}_lr_{}_l2_{}_error_{}_aux_{}_solve_{}_execute_{}_loss_{}.pth".format(
            epoch, args.batch_size, args.seed, args.config, args.img_size,
            args.lr, args.weight_decay, args.error, args.aux, args.solve_mode,
            args.execute_mode, args.loss)
        ckpt_file_path = os.path.join(args.checkpoint_dir, ckpt_model_name)
        torch.save(model.state_dict(), ckpt_file_path)
        model.to(device)

    # save final model
    model.eval()
    model.cpu()
    save_model_name = "Final_epoch_{}_batch_{}_seed_{}_config_{}_img_{}_lr_{}_l2_{}_error_{}_aux_{}_solve_{}_execute_{}_loss_{}.pth".format(
        epoch, args.batch_size, args.seed, args.config, args.img_size, args.lr,
        args.weight_decay, args.error, args.aux, args.solve_mode,
        args.execute_mode, args.loss)
    save_file_path = os.path.join(args.save_dir, save_model_name)
    torch.save(model.state_dict(), save_file_path)

    print("Done. Model saved.")


def test(args, device):

    def test_epoch():
        model.eval()
        correct_sum_avg = 0.0
        xe_sum_avg = 0.0
        total = 0
        test_loader_iter = iter(test_loader)
        spaces = model.update_spaces()
        for _ in trange(len(test_loader_iter)):
            images, targets, _ = next(test_loader_iter)
            total += images.shape[0]
            images = images.to(device)
            targets = targets.to(device)
            panel_attr_probs, attr_inner_objs, attr_pred_probs, object_attr_logprob = model.forward(
                images, *spaces)
            attr_action_probs = model.get_action_probs(panel_attr_probs,
                                                       attr_inner_objs)
            xe_loss, scores = loss_func(panel_attr_probs, attr_pred_probs,
                                        attr_action_probs, error_func, targets)
            xe_sum_avg += xe_loss.item() * images.shape[0]
            correct_sum = criteria.calculate_correct(scores, targets)
            correct_sum_avg += correct_sum
        print("Test Avg Acc: {:.4f}".format(correct_sum_avg / float(total)))
        print("Test Avg XE: {:.4f}".format(xe_sum_avg / float(total)))

    model = get_model(args.config)(DIM_EXIST,
                                   DIM_TYPE,
                                   DIM_SIZE,
                                   DIM_COLOR,
                                   MATRIX_SIZE,
                                   solve_mode=args.solve_mode,
                                   execute_mode=args.execute_mode)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    loss_func = getattr(model, "loss_" + args.loss)
    error_func = getattr(criteria, args.error)
    test_set = Dataset(args.dataset,
                       "test",
                       args.img_size,
                       args.config,
                       test=True)
    test_loader = DataLoader(test_set,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers)
    print("Evaluating on {}".format(args.config))
    with torch.no_grad():
        test_epoch()


def main():
    main_arg_parser = argparse.ArgumentParser(
        description="ALANS, an ALgebra-Aware Neuro-Semi-Symbolic learner")
    subparsers = main_arg_parser.add_subparsers(title="subcommands",
                                                dest="subcommand")

    train_arg_parser = subparsers.add_parser("train",
                                             help="parser for training")
    train_arg_parser.add_argument("--epochs",
                                  type=int,
                                  default=200,
                                  help="the number of training epochs")
    train_arg_parser.add_argument("--batch-size",
                                  type=int,
                                  default=32,
                                  help="size of batch")
    train_arg_parser.add_argument("--seed",
                                  type=int,
                                  default=1234,
                                  help="random number seed")
    train_arg_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="device index for GPU; if GPU unavailable, leave it as default")
    train_arg_parser.add_argument(
        "--dataset",
        type=str,
        default="/home/chizhang/Datasets/RAVEN-10000/",
        help="dataset path")
    train_arg_parser.add_argument("--config",
                                  type=str,
                                  default="distribute_nine",
                                  help="the configuration used for training")
    train_arg_parser.add_argument("--checkpoint-dir",
                                  type=str,
                                  default="./runs/ckpt/",
                                  help="checkpoint save path")
    train_arg_parser.add_argument("--save-dir",
                                  type=str,
                                  default="./runs/save/",
                                  help="final model save path")
    train_arg_parser.add_argument("--log-dir",
                                  type=str,
                                  default="./runs/log/",
                                  help="log save path")
    train_arg_parser.add_argument("--img-size",
                                  type=int,
                                  default=32,
                                  help="image size for training")
    train_arg_parser.add_argument("--lr",
                                  type=float,
                                  default=0.95e-4,
                                  help="learning rate")
    train_arg_parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="weight decay of optimizer, same as l2 reg")
    train_arg_parser.add_argument("--solve-mode",
                                  type=str,
                                  default="mean",
                                  help="solve mode of the inner objective")
    train_arg_parser.add_argument("--execute-mode",
                                  type=str,
                                  default="mean",
                                  help="execute mode of the executor")
    train_arg_parser.add_argument(
        "--error",
        type=str,
        default="JSD",
        help="error used to measure difference between distributions")
    train_arg_parser.add_argument("--num-workers",
                                  type=int,
                                  default=4,
                                  help="number of workers for data loader")
    train_arg_parser.add_argument("--multigpu",
                                  type=int,
                                  default=0,
                                  help="whether to use multi gpu")
    train_arg_parser.add_argument("--resume",
                                  type=str,
                                  default=None,
                                  help="initialized model")
    train_arg_parser.add_argument("--aux",
                                  type=float,
                                  default=1.0,
                                  help="weight of auxiliary training")
    train_arg_parser.add_argument("--loss",
                                  type=str,
                                  choices=["mean", "prob"],
                                  default="mean",
                                  help="final training objective")

    test_arg_parser = subparsers.add_parser("test", help="parser for testing")
    test_arg_parser.add_argument("--batch-size",
                                 type=int,
                                 default=32,
                                 help="size of batch")
    test_arg_parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="device index for GPU; if GPU unavailable, leave it as default")
    test_arg_parser.add_argument("--dataset",
                                 type=str,
                                 default="/home/chizhang/Datasets/RAVEN-10000",
                                 help="dataset path")
    test_arg_parser.add_argument("--config",
                                 type=str,
                                 default="distribute_nine",
                                 help="the configuration used for testing")
    test_arg_parser.add_argument("--model-path",
                                 type=str,
                                 required=True,
                                 help="path to a trained model")
    test_arg_parser.add_argument("--img-size",
                                 type=int,
                                 default=32,
                                 help="image size for training")
    test_arg_parser.add_argument("--solve-mode",
                                 type=str,
                                 choices=["mean", "exact"],
                                 default="mean",
                                 help="solve mode of the inner objective")
    test_arg_parser.add_argument("--execute-mode",
                                 type=str,
                                 choices=["mean", "exact"],
                                 default="mean",
                                 help="execute mode of the executor")
    test_arg_parser.add_argument(
        "--error",
        type=str,
        default="JSD",
        help="error used to measure difference betweeen distributions")
    test_arg_parser.add_argument("--num-workers",
                                 type=int,
                                 default=4,
                                 help="number of workers for data loader")
    test_arg_parser.add_argument("--loss",
                                 type=str,
                                 choices=["mean", "prob"],
                                 default="mean",
                                 help="final training objective")

    args = main_arg_parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda:{}".format(args.device) if args.cuda else "cpu")

    if args.subcommand == "train":
        check_paths(args)
        train(args, device)
    elif args.subcommand == "test":
        test(args, device)
    else:
        print("ERROR: Unknown subcommand")
        sys.exit(1)


if __name__ == "__main__":
    main()
