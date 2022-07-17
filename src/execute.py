# -*- coding: utf-8 -*-

import torch

import utils
from const import PRED_SOFTMAX_SCALE, SOLVE_EPSILON


class Executor(object):
    # "mean" mode puts the expectation inside the squared norm
    # "exact" mode keeps the original form

    def execute_int(self, obs, space, ops, execute_mode="mean"):
        # Execution is inverse solve
        # obs: Tensor of shape (batch, 8, case)
        # space: Tensor of shape (case, matrix_size, matrix_size)
        # ops: List of Tensor of shape (batch, corresponding_matrix_size, corresponding_matrix_size)
        batch, _, case = obs.shape
        _, matrix_size, _ = space.shape
        mean_case = torch.sum(
            obs.reshape(-1, case).unsqueeze(-1).unsqueeze(-1) *
            space.unsqueeze(0),
            dim=1)
        mean_case = mean_case.view(-1, 8, matrix_size, matrix_size)

        # epsilon_matrix_1 = SOLVE_EPSILON * torch.eye(matrix_size, device=obs.device).unsqueeze(0)

        # arity == 1
        # objective: argmin_{M(a_9)} E_{a_8}[||T M(a_8) - M(a_9)||_F^2]
        # solution:  M(a_9) = T E_{a_8}[M(a_8)]
        arity_1_predict = torch.bmm(ops[0], mean_case[:, -1, :, :])
        arity_1_pred_prob = utils.squared_norm(
            arity_1_predict.unsqueeze(1).expand(-1, case, -1, -1) -
            space.unsqueeze(0),
            dim=(2, 3))
        arity_1_pred_prob = torch.softmax(-arity_1_pred_prob *
                                          PRED_SOFTMAX_SCALE,
                                          dim=-1)

        # arity == 2
        # direction == "ltr"
        # objective: argmin_{M(a_9)} E_{a_7, a_8}[||M(a_7) T M(a_8) - M(a_9)||_F^2]
        # solution:  M(a_9) = E_{a_7}[M(a_7)] T E_{a_8}[M(a_8)]
        arity_2_ltr_predict = torch.bmm(
            torch.bmm(mean_case[:, -2, :, :], ops[1]), mean_case[:, -1, :, :])
        arity_2_ltr_pred_prob = utils.squared_norm(
            arity_2_ltr_predict.unsqueeze(1).expand(-1, case, -1, -1) -
            space.unsqueeze(0),
            dim=(2, 3))
        arity_2_ltr_pred_prob = torch.softmax(-arity_2_ltr_pred_prob *
                                              PRED_SOFTMAX_SCALE,
                                              dim=-1)

        # arity == 2
        # direction == "rtl"
        # objective: argmin_{M(a_9)} E_{a_7, a_8}[||M(a_9) T M(a_8) - M(a_7)||_F^2]
        # solution:  M(a_9) T E_{a_8}[M(a_8) M(a_8)^T] T^T = E_{a_7, a_8}[M(a_7) M(a_8)^T] T^T
        if execute_mode == "mean":
            lhs_const = torch.bmm(
                torch.bmm(torch.bmm(ops[2], mean_case[:, -1, :, :]),
                          mean_case[:, -1, :, :].transpose(1, 2)),
                ops[2].transpose(1, 2))
        if execute_mode == "exact":
            square_case = torch.bmm(space, space.transpose(1, 2))
            mean_8_square = torch.sum(
                obs[:, -1, :].unsqueeze(-1).unsqueeze(-1) *
                square_case.unsqueeze(0),
                dim=1)
            lhs_const = torch.bmm(torch.bmm(ops[2], mean_8_square),
                                  ops[2].transpose(1, 2))
        # lhs_const += epsilon_matrix_1
        rhs_const = torch.bmm(
            torch.bmm(mean_case[:, -2, :, :],
                      mean_case[:, -1, :, :].transpose(1, 2)),
            ops[2].transpose(1, 2))
        arity_2_rtl_predict = torch.solve(rhs_const.transpose(1, 2),
                                          lhs_const.transpose(1,
                                                              2))[0].transpose(
                                                                  1, 2)
        arity_2_rtl_pred_prob = utils.squared_norm(
            arity_2_rtl_predict.unsqueeze(1).expand(-1, case, -1, -1) -
            space.unsqueeze(0),
            dim=(2, 3))
        arity_2_rtl_pred_prob = torch.softmax(-arity_2_rtl_pred_prob *
                                              PRED_SOFTMAX_SCALE,
                                              dim=-1)

        # arity == 3
        # direction == "l"
        # objective: argmin_{M(a_9)} E_{a_4, a_6}[||T M(a_6, a_4) - M(a_9)||_F^2]
        # solution:  M(a_9) = T E_{a_4, a_6}[M(a_6, a_4)]
        arity_3_l_predict = torch.bmm(
            ops[3],
            torch.cat([mean_case[:, 5, :, :], mean_case[:, 3, :, :]], dim=1))
        arity_3_l_pred_prob = utils.squared_norm(
            arity_3_l_predict.unsqueeze(1).expand(-1, case, -1, -1) -
            space.unsqueeze(0),
            dim=(2, 3))
        arity_3_l_pred_prob = torch.softmax(-arity_3_l_pred_prob *
                                            PRED_SOFTMAX_SCALE,
                                            dim=-1)

        # arity == 3
        # direction == "r"
        # objective: argmin_{M(a_9)} E_{a_5, a_6}[||T M(a_5, a_6) - M(a_9)||_F^2]
        # solution:  M(a_9) = T E_{a_5, a_6}[M(a_5, a_6)]
        arity_3_r_predict = torch.bmm(
            ops[4],
            torch.cat([mean_case[:, 4, :, :], mean_case[:, 5, :, :]], dim=1))
        arity_3_r_pred_prob = utils.squared_norm(
            arity_3_r_predict.unsqueeze(1).expand(-1, case, -1, -1) -
            space.unsqueeze(0),
            dim=(2, 3))
        arity_3_r_pred_prob = torch.softmax(-arity_3_r_pred_prob *
                                            PRED_SOFTMAX_SCALE,
                                            dim=-1)

        pred_prob = torch.stack([
            arity_1_pred_prob, arity_2_ltr_pred_prob, arity_2_rtl_pred_prob,
            arity_3_l_pred_prob, arity_3_r_pred_prob
        ],
                                dim=1)

        return pred_prob

    def execute_logic(self, obs, space, ops, execute_mode="mean"):
        # Execution is inverse solve
        # obs: Tensor of shape (batch, 8, slot, case)
        # space: Tensor of shape (case, matrix_size, matrix_size)
        # ops: List of Tensor of shape (batch, corresponding_matrix_size, corresponding_matrix_size)
        batch, _, slot, case = obs.shape
        _, matrix_size, _ = space.shape
        mean_case = torch.sum(
            obs.reshape(-1, case).unsqueeze(-1).unsqueeze(-1) *
            space.unsqueeze(0),
            dim=1)
        mean_case = mean_case.view(-1, 8, slot, matrix_size, matrix_size)

        # arity == 1
        # objective: argmin_{M(a_9)} E_pos[E_{a_8}[||T M(a_8^{neigh(pos)}) - M(a_9^pos)||_F^2]]
        # solution:  M(a_9^pos) = T E_{a_8}[M(a_8^{neigh(pos)})] (note pos are independent)
        _, op_row, op_col = ops[0].shape
        arity_1_op = ops[0].unsqueeze(1).expand(-1, slot, -1, -1).reshape(
            -1, op_row, op_col)
        # arity_1_input: Tensor of shape (batch, slot, matrix_size * 3, matrix_size)
        arity_1_input = [
            torch.cat([
                mean_case[:, -1, -1, :, :], mean_case[:, -1, 0, :, :],
                mean_case[:, -1, 1, :, :]
            ],
                      dim=1)
        ]
        for slot_idx in range(1, slot - 1):
            arity_1_input.append(
                mean_case[:, -1, slot_idx - 1:slot_idx + 2, :, :].view(
                    -1, matrix_size * 3, matrix_size))
        arity_1_input.append(
            torch.cat([
                mean_case[:, -1, -2, :, :], mean_case[:, -1, -1, :, :],
                mean_case[:, -1, 0, :, :]
            ],
                      dim=1))
        arity_1_input = torch.stack(arity_1_input, dim=1)
        arity_1_predict = torch.bmm(
            arity_1_op, arity_1_input.view(-1, matrix_size * 3, matrix_size))
        arity_1_pred_prob = utils.squared_norm(
            arity_1_predict.unsqueeze(1).expand(-1, case, -1, -1) -
            space.unsqueeze(0),
            dim=(2, 3))
        arity_1_pred_prob = torch.softmax(
            -arity_1_pred_prob.view(-1, slot, case), dim=-1)

        # arity == 2
        # objective: argmin_{V(a_9)} E_pos[E_{a_7, a_8}[||T V(a_7^pos) \otimes V(a_8^pos) - V(a_9^pos)||_F^2]]
        # solution:  V(a_9^pos) = T E_{a_7}[V(a_7^pos)] \otimes E_{a_8}[V(a_8^pos)]
        _, op_row, op_col = ops[1].shape
        arity_2_op = ops[1].unsqueeze(1).expand(-1, slot, -1, -1).reshape(
            -1, op_row, op_col)
        arity_2_predict = torch.bmm(
            arity_2_op,
            utils.batch_kronecker_product(
                mean_case[:, -2, :, :, :].reshape(-1, matrix_size**2, 1),
                mean_case[:, -1, :, :, :].reshape(-1, matrix_size**2, 1)))
        arity_2_predict = arity_2_predict.view(-1, matrix_size, matrix_size)
        arity_2_pred_prob = utils.squared_norm(
            arity_2_predict.unsqueeze(1).expand(-1, case, -1, -1) -
            space.unsqueeze(0),
            dim=(2, 3))
        arity_2_pred_prob = torch.softmax(
            -arity_2_pred_prob.view(-1, slot, case), dim=-1)

        # arity == 3
        # direction == "l"
        # objective: argmin_{M(a_9)} E_pos[E_{a_4, a_6}[||T M(a_6^pos, a_4^pos) - M(a_9^pos)||_F^2]]
        # solution:  M(a_9^pos) = T E_{a_4, a_6}[M(a_6^pos, a_4^pos)]
        _, op_row, op_col = ops[2].shape
        arity_3_l_op = ops[2].unsqueeze(1).expand(-1, slot, -1, -1).reshape(
            -1, op_row, op_col)
        arity_3_l_input = torch.cat(
            [mean_case[:, 5, :, :, :], mean_case[:, 3, :, :, :]],
            dim=2).view(-1, matrix_size * 2, matrix_size)
        arity_3_l_predict = torch.bmm(arity_3_l_op, arity_3_l_input)
        arity_3_l_pred_prob = utils.squared_norm(
            arity_3_l_predict.unsqueeze(1).expand(-1, case, -1, -1) -
            space.unsqueeze(0),
            dim=(2, 3))
        arity_3_l_pred_prob = torch.softmax(
            -arity_3_l_pred_prob.view(-1, slot, case), dim=-1)

        # arity == 3
        # direction == "r"
        # objective: argmin_{M(a_9)} E_pos[E_{a_5, a_6}[||T M(a_5^pos, a_6^pos) - M(a_9^pos)||_F^2]]
        # solution:  M(a_9^pos) = T E_{a_5, a_6}[M(a_5^pos, a_6^pos)]
        _, op_row, op_col = ops[3].shape
        arity_3_r_op = ops[3].unsqueeze(1).expand(-1, slot, -1, -1).reshape(
            -1, op_row, op_col)
        arity_3_r_input = torch.cat(
            [mean_case[:, 4, :, :, :], mean_case[:, 5, :, :, :]],
            dim=2).view(-1, matrix_size * 2, matrix_size)
        arity_3_r_predict = torch.bmm(arity_3_r_op, arity_3_r_input)
        arity_3_r_pred_prob = utils.squared_norm(
            arity_3_r_predict.unsqueeze(1).expand(-1, case, -1, -1) -
            space.unsqueeze(0),
            dim=(2, 3))
        arity_3_r_pred_prob = torch.softmax(
            -arity_3_r_pred_prob.view(-1, slot, case), dim=-1)

        pred_prob = torch.stack([
            arity_1_pred_prob, arity_2_pred_prob, arity_3_l_pred_prob,
            arity_3_r_pred_prob
        ],
                                dim=1)

        return pred_prob
