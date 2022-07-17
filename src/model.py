# -*- coding: utf-8 -*-

import torch

from attr import Attribute, NumberSys
from const import ACTION_SOFTMAX_SCALE, ANSWER_SOFTMAX_SCALE, INNER_REG
from execute import Executor
from inner import InnerObj
from network import ObjectCNN
from scene import SceneEngine
from utils import log, rgetattr


class Model(object):

    def __init__(self,
                 name,
                 number_dims,
                 exist_dim,
                 type_dim,
                 size_dim,
                 color_dim,
                 matrix_size,
                 solve_mode="mean",
                 execute_mode="mean"):
        # number_dims: List of numbers in each possible component
        # bookkeeping
        int_size = max(number_dims + [type_dim, size_dim, color_dim])
        logic_size = exist_dim
        self.number_dims = number_dims
        self.matrix_size = matrix_size
        self.solve_mode = solve_mode
        self.execute_mode = execute_mode

        # initialize analogical representation
        self.number_sys = NumberSys(matrix_size, int_size, logic_size)
        self.exist = Attribute("exist", exist_dim, "logic")
        self.numbers = [
            Attribute("number", number_dim, "int")
            for number_dim in number_dims
        ]
        self.type = Attribute("type", type_dim, "int")
        self.size = Attribute("size", size_dim, "int")
        self.color = Attribute("color", color_dim, "int")

        # initialize perception models
        self.object_cnn = ObjectCNN(exist_dim, type_dim, size_dim, color_dim)
        self.scene_engines = [
            SceneEngine(number_dim) for number_dim in number_dims
        ]

        # initialize reasoning models
        self.inner_obj = InnerObj()
        self.executor = Executor()

    def to(self, device):
        self.number_sys = self.number_sys.to(device)
        self.object_cnn = self.object_cnn.to(device)
        for i in range(len(self.scene_engines)):
            self.scene_engines[i] = self.scene_engines[i].to(device)

    def state_dict(self):
        return {
            "number_sys": self.number_sys.state_dict(),
            "object_cnn": self.object_cnn.state_dict()
        }

    def load_state_dict(self, ckpt):
        if "number_sys" in ckpt:
            self.number_sys.load_state_dict(ckpt["number_sys"], strict=False)
        if "object_cnn" in ckpt:
            self.object_cnn.load_state_dict(ckpt["object_cnn"])

    def parallelize(self):
        self.number_sys = torch.nn.DataParallel(self.number_sys)
        self.object_cnn = torch.nn.DataParallel(self.object_cnn)
        for i in range(len(self.scene_engines)):
            self.scene_engines[i] = torch.nn.DataParallel(
                self.scene_engines[i])

    def mute_module(self, module_name):
        module = rgetattr(self, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = False

    def parameters(self):
        for param_module in [self.number_sys, self.object_cnn]:
            for param in param_module.parameters():
                yield param

    def train(self, mode=True):
        self.number_sys.train(mode)
        self.object_cnn.train(mode)
        for i in range(len(self.scene_engines)):
            self.scene_engines[i].train(mode)

    def eval(self):
        self.train(False)

    def cpu(self):
        self.number_sys.cpu()
        self.object_cnn.cpu()
        for i in range(len(self.scene_engines)):
            self.scene_engines[i].cpu()

    def cuda(self):
        self.number_sys.cuda()
        self.object_cnn.cuda()
        for i in range(len(self.scene_engines)):
            self.scene_engines[i].cuda()

    def infer_scene(self, object_attr_logprob):
        "Implement for different Models"
        return []

    def get_object_attr_logprob(self, x):
        return self.object_cnn(x)

    def update_spaces(self):
        # prepare number system representation
        # update_spaces should be called every time you update the model
        # i.e., call it every time before forward during training
        # but call it only once during testing
        int_space, logic_space = self.number_sys()
        exist_space = self.exist.construct(int_space, logic_space)
        number_spaces = [
            number.construct(int_space, logic_space) for number in self.numbers
        ]
        type_space = self.type.construct(int_space, logic_space)
        size_space = self.size.construct(int_space, logic_space)
        color_space = self.color.construct(int_space, logic_space)
        return exist_space, number_spaces, type_space, size_space, color_space

    def forward(self, x, exist_space, number_spaces, type_space, size_space,
                color_space):
        # perception
        object_attr_logprob = self.object_cnn(x)
        panel_attr_probs = self.infer_scene(object_attr_logprob)

        attr_inner_objs = []
        attr_pred_probs = []
        for idx in range(len(self.number_dims)):
            # reason per component
            number_space = number_spaces[idx]
            panel_attr_prob = panel_attr_probs[idx]

            def reason_component(panel_attr_prob, number_space, include_exist):
                exist_prob = panel_attr_prob.exist_prob
                number_prob = panel_attr_prob.number_prob
                type_prob = panel_attr_prob.norm_type_prob
                size_prob = panel_attr_prob.norm_size_prob
                color_prob = panel_attr_prob.norm_color_prob

                if include_exist:
                    # reasoning on exist
                    exist_inner_objs, exist_ops = self.inner_obj.solve_logic(
                        exist_prob[:, :8, :, :], exist_space, INNER_REG,
                        self.solve_mode)
                    exist_pred_prob = self.executor.execute_logic(
                        exist_prob[:, :8, :, :], exist_space, exist_ops,
                        self.execute_mode)

                    # reasoning on number
                    number_inner_objs, number_ops = self.inner_obj.solve_int(
                        number_prob[:, :8, :], number_space, INNER_REG * 0.01,
                        self.solve_mode)
                    number_pred_prob = self.executor.execute_int(
                        number_prob[:, :8, :], number_space, number_ops,
                        self.execute_mode)
                else:
                    exist_inner_objs = None
                    number_inner_objs = None
                    exist_pred_prob = None
                    number_pred_prob = None

                # reasoning on type
                type_inner_objs, type_ops = self.inner_obj.solve_int(
                    type_prob[:, :8, :], type_space, INNER_REG * 0.01,
                    self.solve_mode)
                type_pred_prob = self.executor.execute_int(
                    type_prob[:, :8, :], type_space, type_ops,
                    self.execute_mode)

                # reasoning on size
                size_inner_objs, size_ops = self.inner_obj.solve_int(
                    size_prob[:, :8, :], size_space, INNER_REG * 0.01,
                    self.solve_mode)
                size_pred_prob = self.executor.execute_int(
                    size_prob[:, :8, :], size_space, size_ops,
                    self.execute_mode)

                # reasoning on color
                color_inner_objs, color_ops = self.inner_obj.solve_int(
                    color_prob[:, :8, :], color_space, INNER_REG * 0.005,
                    self.solve_mode)
                color_pred_prob = self.executor.execute_int(
                    color_prob[:, :8, :], color_space, color_ops,
                    self.execute_mode)

                return [exist_inner_objs, number_inner_objs, type_inner_objs, size_inner_objs, color_inner_objs], \
                       [exist_pred_prob, number_pred_prob, type_pred_prob, size_pred_prob, color_pred_prob],

            attr_inner_obj, attr_pred_prob = reason_component(
                panel_attr_prob, number_space, self.number_dims[idx] > 1)
            attr_inner_objs.append(attr_inner_obj)
            attr_pred_probs.append(attr_pred_prob)

        return panel_attr_probs, attr_inner_objs, attr_pred_probs, object_attr_logprob

    def get_action_probs(self, panel_attr_probs, attr_inner_objs):
        attr_action_probs = []
        for idx in range(len(self.number_dims)):
            exist_inner_objs, number_inner_objs, type_inner_objs, size_inner_objs, color_inner_objs = attr_inner_objs[
                idx]
            if self.number_dims[idx] > 1:
                exist_action_prob = torch.softmax(-exist_inner_objs, dim=-1)
                number_action_prob = torch.softmax(-number_inner_objs *
                                                   ACTION_SOFTMAX_SCALE * 1e-5,
                                                   dim=-1)
                exist_number_select = torch.softmax(-torch.stack([
                    torch.min(exist_inner_objs, dim=-1)[0],
                    torch.min(number_inner_objs, dim=-1)[0]
                ],
                                                                 dim=-1),
                                                    dim=-1)
            else:
                exist_action_prob = None
                number_action_prob = None
                exist_number_select = None
            rule_type_prob = panel_attr_probs[idx].rule_type_prob.unsqueeze(-1)
            type_action_prob = torch.softmax(-type_inner_objs *
                                             ACTION_SOFTMAX_SCALE,
                                             dim=-1)
            type_action_prob = torch.cat(
                [type_action_prob * rule_type_prob, 1.0 - rule_type_prob],
                dim=-1)
            rule_size_prob = panel_attr_probs[idx].rule_size_prob.unsqueeze(-1)
            size_action_prob = torch.softmax(-size_inner_objs *
                                             ACTION_SOFTMAX_SCALE,
                                             dim=-1)
            size_action_prob = torch.cat(
                [size_action_prob * rule_size_prob, 1.0 - rule_size_prob],
                dim=-1)
            rule_color_prob = panel_attr_probs[idx].rule_color_prob.unsqueeze(
                -1)
            color_action_prob = torch.softmax(-color_inner_objs *
                                              ACTION_SOFTMAX_SCALE,
                                              dim=-1)
            color_action_prob = torch.cat(
                [color_action_prob * rule_color_prob, 1.0 - rule_color_prob],
                dim=-1)
            attr_action_probs.append([
                exist_action_prob, number_action_prob, exist_number_select,
                type_action_prob, size_action_prob, color_action_prob
            ])
        return attr_action_probs

    def loss_prob(self, panel_attr_probs, attr_pred_probs, attr_action_probs,
                  error_func, targets):
        scores = 0.0
        for idx in range(len(self.number_dims)):
            exist_prob = panel_attr_probs[idx].exist_prob
            number_prob = panel_attr_probs[idx].number_prob
            type_prob = panel_attr_probs[idx].type_prob
            size_prob = panel_attr_probs[idx].size_prob
            color_prob = panel_attr_probs[idx].color_prob
            exist_pred_prob, number_pred_prob, type_pred_prob, size_pred_prob, color_pred_prob = attr_pred_probs[
                idx]
            exist_action_prob, number_action_prob, exist_number_select, type_action_prob, size_action_prob, color_action_prob = attr_action_probs[
                idx]

            batch = exist_prob.shape[0]
            zeros = torch.zeros((batch, 5, 1), device=exist_prob.device)
            uniform = torch.ones((batch, 8, 1), device=exist_prob.device) / 8.0

            if self.number_dims[idx] > 1:
                all_exist_error = torch.mean(error_func(
                    exist_pred_prob.unsqueeze(1).expand(-1, 8, -1, -1, -1),
                    exist_prob[:, 8:, :, :].unsqueeze(2).expand(
                        -1, -1, 4, -1, -1)),
                                             dim=-1)
                exist_answer_prob = torch.softmax(-all_exist_error, dim=1)
                exist_answer_prob = torch.sum(exist_answer_prob *
                                              exist_action_prob.unsqueeze(1),
                                              dim=-1)
                all_number_error = error_func(
                    number_pred_prob.unsqueeze(1).expand(-1, 8, -1, -1),
                    number_prob[:, 8:, :].unsqueeze(2).expand(-1, -1, 5, -1))
                number_answer_prob = torch.softmax(-all_number_error *
                                                   ANSWER_SOFTMAX_SCALE,
                                                   dim=1)
                number_answer_prob = torch.sum(number_answer_prob *
                                               number_action_prob.unsqueeze(1),
                                               dim=-1)
                exist_number_answer_prob = exist_answer_prob * exist_number_select[:, :
                                                                                   1] + number_answer_prob * exist_number_select[:,
                                                                                                                                 1:
                                                                                                                                 2]
                scores += log(exist_number_answer_prob)
            type_pred_prob = torch.cat([type_pred_prob, zeros], dim=-1)
            all_type_error = error_func(
                type_pred_prob.unsqueeze(1).expand(-1, 8, -1, -1),
                type_prob[:, 8:, :].unsqueeze(2).expand(-1, -1, 5, -1))
            type_answer_prob = torch.softmax(-all_type_error *
                                             ANSWER_SOFTMAX_SCALE,
                                             dim=1)
            type_answer_prob = torch.cat([type_answer_prob, uniform], dim=-1)
            type_answer_prob = torch.sum(type_answer_prob *
                                         type_action_prob.unsqueeze(1),
                                         dim=-1)
            scores += log(type_answer_prob)
            size_pred_prob = torch.cat([size_pred_prob, zeros], dim=-1)
            all_size_error = error_func(
                size_pred_prob.unsqueeze(1).expand(-1, 8, -1, -1),
                size_prob[:, 8:, :].unsqueeze(2).expand(-1, -1, 5, -1))
            size_answer_prob = torch.softmax(-all_size_error *
                                             ANSWER_SOFTMAX_SCALE,
                                             dim=1)
            size_answer_prob = torch.cat([size_answer_prob, uniform], dim=-1)
            size_answer_prob = torch.sum(size_answer_prob *
                                         size_action_prob.unsqueeze(1),
                                         dim=-1)
            scores += log(size_answer_prob)
            color_pred_prob = torch.cat([color_pred_prob, zeros], dim=-1)
            all_color_error = error_func(
                color_pred_prob.unsqueeze(1).expand(-1, 8, -1, -1),
                color_prob[:, 8:, :].unsqueeze(2).expand(-1, -1, 5, -1))
            color_answer_prob = torch.softmax(-all_color_error *
                                              ANSWER_SOFTMAX_SCALE,
                                              dim=1)
            color_answer_prob = torch.cat([color_answer_prob, uniform], dim=-1)
            color_answer_prob = torch.sum(color_answer_prob *
                                          color_action_prob.unsqueeze(1),
                                          dim=-1)
            scores += log(color_answer_prob)
        negative_reward = torch.nn.functional.cross_entropy(scores, targets)
        return negative_reward, scores

    def loss_mean(self, panel_attr_probs, attr_pred_probs, attr_action_probs,
                  error_func, targets):
        dist = 0.0
        for idx in range(len(self.number_dims)):
            exist_prob = panel_attr_probs[idx].exist_prob
            number_prob = panel_attr_probs[idx].number_prob
            type_prob = panel_attr_probs[idx].type_prob
            size_prob = panel_attr_probs[idx].size_prob
            color_prob = panel_attr_probs[idx].color_prob
            exist_pred_prob, number_pred_prob, type_pred_prob, size_pred_prob, color_pred_prob = attr_pred_probs[
                idx]
            exist_action_prob, number_action_prob, exist_number_select, type_action_prob, size_action_prob, color_action_prob = attr_action_probs[
                idx]

            batch = exist_prob.shape[0]
            zeros = torch.zeros((batch, 5, 1), device=exist_prob.device)

            if self.number_dims[idx] > 1:
                all_exist_error = torch.mean(error_func(
                    exist_pred_prob.unsqueeze(1).expand(-1, 8, -1, -1, -1),
                    exist_prob[:, 8:, :, :].unsqueeze(2).expand(
                        -1, -1, 4, -1, -1)),
                                             dim=-1)
                all_number_error = error_func(
                    number_pred_prob.unsqueeze(1).expand(-1, 8, -1, -1),
                    number_prob[:, 8:, :].unsqueeze(2).expand(-1, -1, 5, -1))
                exist_mean_dist = torch.sum(
                    exist_action_prob.unsqueeze(1) * all_exist_error,
                    dim=-1) * exist_number_select[:, :1]
                number_mean_dist = torch.sum(
                    number_action_prob.unsqueeze(1) * all_number_error,
                    dim=-1) * exist_number_select[:, 1:2]
                dist += exist_mean_dist + number_mean_dist
            type_pred_prob = torch.cat([type_pred_prob, zeros], dim=-1)
            all_type_error = error_func(
                type_pred_prob.unsqueeze(1).expand(-1, 8, -1, -1),
                type_prob[:, 8:, :].unsqueeze(2).expand(-1, -1, 5, -1))
            type_mean_dist = torch.sum(type_action_prob[:, :-1].unsqueeze(1) *
                                       all_type_error,
                                       dim=-1)
            dist += type_mean_dist
            size_pred_prob = torch.cat([size_pred_prob, zeros], dim=-1)
            all_size_error = error_func(
                size_pred_prob.unsqueeze(1).expand(-1, 8, -1, -1),
                size_prob[:, 8:, :].unsqueeze(2).expand(-1, -1, 5, -1))
            size_mean_dist = torch.sum(size_action_prob[:, :-1].unsqueeze(1) *
                                       all_size_error,
                                       dim=-1)
            dist += size_mean_dist
            color_pred_prob = torch.cat([color_pred_prob, zeros], dim=-1)
            all_color_error = error_func(
                color_pred_prob.unsqueeze(1).expand(-1, 8, -1, -1),
                color_prob[:, 8:, :].unsqueeze(2).expand(-1, -1, 5, -1))
            color_mean_dist = torch.sum(
                color_action_prob[:, :-1].unsqueeze(1) * all_color_error,
                dim=-1)
            dist += color_mean_dist
        scores = -dist
        negative_reward = torch.nn.functional.cross_entropy(scores, targets)
        return negative_reward, scores

    def action_auxiliary_loss(self, attr_action_probs, attr_action_targets):
        exist_loss = 0.0
        number_loss = 0.0
        type_loss = 0.0
        size_loss = 0.0
        color_loss = 0.0
        for idx in range(len(self.number_dims)):
            exist_action_prob, number_action_prob, _, type_action_prob, size_action_prob, color_action_prob = attr_action_probs[
                idx]
            exist_indicator, number_indicator, exist_gt, number_gt, type_gt, size_gt, color_gt = attr_action_targets[
                idx]
            if self.number_dims[idx] > 1:
                total_exist = torch.sum(exist_indicator)
                total_number = torch.sum(number_indicator)
                exist_before_reduce = -log(
                    torch.gather(exist_action_prob, -1, exist_gt.unsqueeze(
                        -1)).squeeze(-1)) * exist_indicator.float()
                exist_loss += exist_before_reduce.sum() / total_exist
                number_before_reduce = -log(
                    torch.gather(number_action_prob, -1, number_gt.unsqueeze(
                        -1)).squeeze(-1)) * number_indicator.float()
                number_loss += number_before_reduce.sum() / total_number
            type_loss += -log(
                torch.gather(type_action_prob, -1,
                             type_gt.unsqueeze(-1)).squeeze(-1)).mean()
            size_loss += -log(
                torch.gather(size_action_prob, -1,
                             size_gt.unsqueeze(-1)).squeeze(-1)).mean()
            color_loss += -log(
                torch.gather(color_action_prob, -1,
                             color_gt.unsqueeze(-1)).squeeze(-1)).mean()
        return exist_loss, number_loss, type_loss, size_loss, color_loss


class CenterSingle(Model):

    def __init__(self,
                 exist_dim,
                 type_dim,
                 size_dim,
                 color_dim,
                 matrix_size,
                 solve_mode="mean",
                 execute_mode="mean"):
        super(CenterSingle, self).__init__("CenterSingle", [1],
                                           exist_dim,
                                           type_dim,
                                           size_dim,
                                           color_dim,
                                           matrix_size,
                                           solve_mode=solve_mode,
                                           execute_mode=execute_mode)

    def infer_scene(self, object_attr_logprob):
        "Implement for different Models"
        panel_attr_prob = self.scene_engines[0].compute_scene_prob(
            *object_attr_logprob)
        return [panel_attr_prob]


class DistributeFour(Model):

    def __init__(self,
                 exist_dim,
                 type_dim,
                 size_dim,
                 color_dim,
                 matrix_size,
                 solve_mode="mean",
                 execute_mode="mean"):
        super(DistributeFour, self).__init__("DistributeFour", [4],
                                             exist_dim,
                                             type_dim,
                                             size_dim,
                                             color_dim,
                                             matrix_size,
                                             solve_mode=solve_mode,
                                             execute_mode=execute_mode)

    def infer_scene(self, object_attr_logprob):
        "Implement for different Models"
        panel_attr_prob = self.scene_engines[0].compute_scene_prob(
            *object_attr_logprob)
        return [panel_attr_prob]


class DistributeNine(Model):

    def __init__(self,
                 exist_dim,
                 type_dim,
                 size_dim,
                 color_dim,
                 matrix_size,
                 solve_mode="mean",
                 execute_mode="mean"):
        super(DistributeNine, self).__init__("DistributeNine", [9],
                                             exist_dim,
                                             type_dim,
                                             size_dim,
                                             color_dim,
                                             matrix_size,
                                             solve_mode=solve_mode,
                                             execute_mode=execute_mode)

    def infer_scene(self, object_attr_logprob):
        "Implement for different Models"
        panel_attr_prob = self.scene_engines[0].compute_scene_prob(
            *object_attr_logprob)
        return [panel_attr_prob]


class InCenterSingleOutCenterSingle(Model):

    def __init__(self,
                 exist_dim,
                 type_dim,
                 size_dim,
                 color_dim,
                 matrix_size,
                 solve_mode="mean",
                 execute_mode="mean"):
        super(InCenterSingleOutCenterSingle,
              self).__init__("InCenterSingleOutCenterSingle", [1, 1],
                             exist_dim,
                             type_dim,
                             size_dim,
                             color_dim,
                             matrix_size,
                             solve_mode=solve_mode,
                             execute_mode=execute_mode)

    def infer_scene(self, object_attr_logprob):
        "Implement for different Models"
        in_panel_attr_prob = self.scene_engines[0].compute_scene_prob(
            object_attr_logprob[0][:, :, :1, :],
            object_attr_logprob[1][:, :, :1, :],
            object_attr_logprob[2][:, :, :1, :],
            object_attr_logprob[3][:, :, :1, :])
        out_panel_attr_prob = self.scene_engines[1].compute_scene_prob(
            object_attr_logprob[0][:, :, 1:, :], object_attr_logprob[1][:, :,
                                                                        1:, :],
            object_attr_logprob[2][:, :, 1:, :], object_attr_logprob[3][:, :,
                                                                        1:, :])
        return [in_panel_attr_prob, out_panel_attr_prob]


class InDistributeFourOutCenterSingle(Model):

    def __init__(self,
                 exist_dim,
                 type_dim,
                 size_dim,
                 color_dim,
                 matrix_size,
                 solve_mode="mean",
                 execute_mode="mean"):
        super(InDistributeFourOutCenterSingle,
              self).__init__("InDistributeFourOutCenterSingle", [4, 1],
                             exist_dim,
                             type_dim,
                             size_dim,
                             color_dim,
                             matrix_size,
                             solve_mode=solve_mode,
                             execute_mode=execute_mode)

    def infer_scene(self, object_attr_logprob):
        "Implement for different Models"
        in_panel_attr_prob = self.scene_engines[0].compute_scene_prob(
            object_attr_logprob[0][:, :, :4, :],
            object_attr_logprob[1][:, :, :4, :],
            object_attr_logprob[2][:, :, :4, :],
            object_attr_logprob[3][:, :, :4, :])
        out_panel_attr_prob = self.scene_engines[1].compute_scene_prob(
            object_attr_logprob[0][:, :, 4:, :], object_attr_logprob[1][:, :,
                                                                        4:, :],
            object_attr_logprob[2][:, :, 4:, :], object_attr_logprob[3][:, :,
                                                                        4:, :])
        return [in_panel_attr_prob, out_panel_attr_prob]


class LeftCenterSingleRightCenterSingle(Model):

    def __init__(self,
                 exist_dim,
                 type_dim,
                 size_dim,
                 color_dim,
                 matrix_size,
                 solve_mode="mean",
                 execute_mode="mean"):
        super(LeftCenterSingleRightCenterSingle,
              self).__init__("LeftCenterSingleRightCenterSingle", [1, 1],
                             exist_dim,
                             type_dim,
                             size_dim,
                             color_dim,
                             matrix_size,
                             solve_mode=solve_mode,
                             execute_mode=execute_mode)

    def infer_scene(self, object_attr_logprob):
        "Implement for different Models"
        left_panel_attr_prob = self.scene_engines[0].compute_scene_prob(
            object_attr_logprob[0][:, :, :1, :],
            object_attr_logprob[1][:, :, :1, :],
            object_attr_logprob[2][:, :, :1, :],
            object_attr_logprob[3][:, :, :1, :])
        right_panel_attr_prob = self.scene_engines[1].compute_scene_prob(
            object_attr_logprob[0][:, :, 1:, :], object_attr_logprob[1][:, :,
                                                                        1:, :],
            object_attr_logprob[2][:, :, 1:, :], object_attr_logprob[3][:, :,
                                                                        1:, :])
        return [left_panel_attr_prob, right_panel_attr_prob]


class UpCenterSingleDownCenterSingle(Model):

    def __init__(self,
                 exist_dim,
                 type_dim,
                 size_dim,
                 color_dim,
                 matrix_size,
                 solve_mode="mean",
                 execute_mode="mean"):
        super(UpCenterSingleDownCenterSingle,
              self).__init__("UpCenterSingleDownCenterSingle", [1, 1],
                             exist_dim,
                             type_dim,
                             size_dim,
                             color_dim,
                             matrix_size,
                             solve_mode=solve_mode,
                             execute_mode=execute_mode)

    def infer_scene(self, object_attr_logprob):
        "Implement for different Models"
        up_panel_attr_prob = self.scene_engines[0].compute_scene_prob(
            object_attr_logprob[0][:, :, :1, :],
            object_attr_logprob[1][:, :, :1, :],
            object_attr_logprob[2][:, :, :1, :],
            object_attr_logprob[3][:, :, :1, :])
        down_panel_attr_prob = self.scene_engines[1].compute_scene_prob(
            object_attr_logprob[0][:, :, 1:, :], object_attr_logprob[1][:, :,
                                                                        1:, :],
            object_attr_logprob[2][:, :, 1:, :], object_attr_logprob[3][:, :,
                                                                        1:, :])
        return [up_panel_attr_prob, down_panel_attr_prob]
