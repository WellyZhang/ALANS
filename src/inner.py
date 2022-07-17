# -*- coding: utf-8 -*-

import torch

from utils import batch_kronecker_product, squared_norm


class InnerObj(object):
    # "mean" mode puts the expectation inside the squared norm
    # "exact" mode keeps the original form
    # obj is approximated by putting expectation inside

    def solve_int(self, obs, space, reg, solve_mode="mean"):
        # obs: Tensor of shape (batch, 8, case)
        # space: Tensor of shape (case, matrix_size, matrix_size)
        batch, _, case = obs.shape
        _, matrix_size, _ = space.shape
        mean_case = torch.sum(
            obs.reshape(-1, case).unsqueeze(-1).unsqueeze(-1) *
            space.unsqueeze(0),
            dim=1)
        mean_case = mean_case.view(-1, 8, matrix_size, matrix_size)

        # arity == 1
        # inner obj is:
        #     1/5 (E_{a_1, a_2}[||T M(a_1) - M(a_2)||_F^2] +
        #          E_{a_2, a_3}[||T M(a_2) - M(a_3)||_F^2] +
        #          E_{a_4, a_5}[||T M(a_4) - M(a_5)||_F^2] +
        #          E_{a_5, a_6}[||T M(a_5) - M(a_6)||_F^2] +
        #          E_{a_7, a_8}[||T M(a_7) - M(a_8)||_F^2]) + lambda * ||T||_F^2
        # solution is:
        #     T(E_{a_1}[M(a_1) M(a_1)^T] +
        #       E_{a_2}[M(a_2) M(a_2)^T] +
        #       E_{a_4}[M(a_4) M(a_4)^T] +
        #       E_{a_5}[M(a_5) M(a_5)^T] +
        #       E_{a_7}[M(a_7) M(a_7)^T] + 5 * lambda * I) =
        #     (E_{a_1, a_2}[M(a_2) M(a_1)^T] +
        #      E_{a_2, a_3}[M(a_3) M(a_2)^T] +
        #      E_{a_4, a_5}[M(a_5) M(a_4)^T] +
        #      E_{a_5, a_6}[M(a_6) M(a_5)^T] +
        #      E_{a_7, a_8}[M(a_8) M(a_7)^T])
        if solve_mode == "mean":
            arity_1_lhs_const = torch.bmm(mean_case[:, 0, :, :], mean_case[:, 0, :, :].transpose(1, 2)) + \
                                torch.bmm(mean_case[:, 1, :, :], mean_case[:, 1, :, :].transpose(1, 2)) + \
                                torch.bmm(mean_case[:, 3, :, :], mean_case[:, 3, :, :].transpose(1, 2)) + \
                                torch.bmm(mean_case[:, 4, :, :], mean_case[:, 4, :, :].transpose(1, 2)) + \
                                torch.bmm(mean_case[:, 6, :, :], mean_case[:, 6, :, :].transpose(1, 2))
        if solve_mode == "exact":
            mean_right_square_case = torch.sum(
                obs.reshape(-1, case).unsqueeze(-1).unsqueeze(-1) *
                torch.bmm(space, space.transpose(1, 2)).unsqueeze(0),
                dim=1)
            mean_right_square_case = mean_right_square_case.view(
                -1, 8, matrix_size, matrix_size)
            arity_1_lhs_const = mean_right_square_case[:, 0, :, :] + \
                                mean_right_square_case[:, 1, :, :] + \
                                mean_right_square_case[:, 3, :, :] + \
                                mean_right_square_case[:, 4, :, :] + \
                                mean_right_square_case[:, 6, :, :]
        arity_1_lhs_const += 5 * reg * torch.eye(
            matrix_size, device=obs.device).unsqueeze(0)
        arity_1_rhs_const = torch.bmm(mean_case[:, 1, :, :], mean_case[:, 0, :, :].transpose(1, 2)) + \
                            torch.bmm(mean_case[:, 2, :, :], mean_case[:, 1, :, :].transpose(1, 2)) + \
                            torch.bmm(mean_case[:, 4, :, :], mean_case[:, 3, :, :].transpose(1, 2)) + \
                            torch.bmm(mean_case[:, 5, :, :], mean_case[:, 4, :, :].transpose(1, 2)) + \
                            torch.bmm(mean_case[:, 7, :, :], mean_case[:, 6, :, :].transpose(1, 2))
        arity_1_ops = torch.solve(arity_1_rhs_const.transpose(1, 2),
                                  arity_1_lhs_const.transpose(1,
                                                              2))[0].transpose(
                                                                  1, 2)
        arity_1_inner_objs = 1.0 / 5 * (squared_norm(torch.bmm(arity_1_ops, mean_case[:, 0, :, :]) - mean_case[:, 1, :, :], dim=(1, 2)) +
                                        squared_norm(torch.bmm(arity_1_ops, mean_case[:, 1, :, :]) - mean_case[:, 2, :, :], dim=(1, 2)) +
                                        squared_norm(torch.bmm(arity_1_ops, mean_case[:, 3, :, :]) - mean_case[:, 4, :, :], dim=(1, 2)) +
                                        squared_norm(torch.bmm(arity_1_ops, mean_case[:, 4, :, :]) - mean_case[:, 5, :, :], dim=(1, 2)) +
                                        squared_norm(torch.bmm(arity_1_ops, mean_case[:, 6, :, :]) - mean_case[:, 7, :, :], dim=(1, 2))) + \
                             reg * squared_norm(arity_1_ops, dim=(1, 2))

        # arity == 2
        # direction: left to right
        # inner obj is:
        #     1/2 (E_{a_1, a_2, a_3}[||M(a_1) T M(a_2) - M(a_3)||_F^2] +
        #          E_{a_4, a_5, a_6}[||M(a_4) T M(a_5) - M(a_6)||_F^2]) + lambda * ||T||_F^2
        # solution is:
        #     (E_{a_1}[M(a_1)^T M(a_1)] T E_{a_2}[M(a_2) M(a_2)^T] +
        #      E_{a_4}[M(a_4)^T M(a_4)] T E_{a_5}[M(a_5) M(a_5)^T]) + 2 * lambda * T =
        #     (E_{a_1, a_2, a_3}[M(a_1)^T M(a_3) M(a_2)^T] +
        #      E_{a_4, a_5, a_6}[M(a_4)^T M(a_6) M(a_5)^T])
        # Note that this equation is not a linear equation but a linear matrix equation of \sum_k A_k T B_k = C
        # to solve it, note that vec(A T B) = A \otimes B^T vec(T), where \otimes is the matrix kronecker product
        # and vec(A + B) = vec(A) + vec(B); applying the operator to both sides turns it into linear system
        # ref: Explicit Solutions of Linear Matrix Equations
        if solve_mode == "mean":
            arity_2_ltr_lhs_const = batch_kronecker_product(torch.bmm(mean_case[:, 0, :, :].transpose(1, 2), mean_case[:, 0, :, :]),
                                                            torch.bmm(mean_case[:, 1, :, :], mean_case[:, 1, :, :].transpose(1, 2))) + \
                                    batch_kronecker_product(torch.bmm(mean_case[:, 3, :, :].transpose(1, 2), mean_case[:, 3, :, :]),
                                                            torch.bmm(mean_case[:, 4, :, :], mean_case[:, 4, :, :].transpose(1, 2)))
        if solve_mode == "exact":
            mean_left_square_case = torch.sum(
                obs.reshape(-1, case).unsqueeze(-1).unsqueeze(-1) *
                torch.bmm(space.transpose(1, 2), space).unsqueeze(0),
                dim=1)
            mean_left_square_case = mean_left_square_case.view(
                -1, 8, matrix_size, matrix_size)
            arity_2_ltr_lhs_const = batch_kronecker_product(mean_left_square_case[:, 0, :, :], mean_right_square_case[:, 1, :, :]) + \
                                    batch_kronecker_product(mean_left_square_case[:, 3, :, :], mean_right_square_case[:, 4, :, :])
        arity_2_ltr_lhs_const += 2 * reg * torch.eye(
            matrix_size**2, device=obs.device).unsqueeze(0)
        arity_2_ltr_rhs_const = torch.bmm(torch.bmm(mean_case[:, 0, :, :].transpose(1, 2), mean_case[:, 2, :, :]),
                                          mean_case[:, 1, :, :].transpose(1, 2)) + \
                                torch.bmm(torch.bmm(mean_case[:, 3, :, :].transpose(1, 2), mean_case[:, 5, :, :]),
                                          mean_case[:, 4, :, :].transpose(1, 2))
        arity_2_ltr_rhs_const = arity_2_ltr_rhs_const.view(
            -1, matrix_size * matrix_size, 1)
        arity_2_ltr_ops = torch.solve(arity_2_ltr_rhs_const,
                                      arity_2_ltr_lhs_const)[0].view(
                                          -1, matrix_size, matrix_size)
        arity_2_ltr_inner_objs = 1.0 / 2 * (squared_norm(torch.bmm(torch.bmm(mean_case[:, 0, :, :], arity_2_ltr_ops),
                                                                   mean_case[:, 1, :, :]) - mean_case[:, 2, :, :], dim=(1, 2)) +
                                            squared_norm(torch.bmm(torch.bmm(mean_case[:, 3, :, :], arity_2_ltr_ops),
                                                                   mean_case[:, 4, :, :]) - mean_case[:, 5, :, :], dim=(1, 2))) + \
                                 reg * squared_norm(arity_2_ltr_ops, dim=(1, 2))

        # arity == 2
        # direction: right to left
        # inner obj is:
        #     1/2 (E_{a_1, a_2, a_3}[||M(a_3) T M(a_2) - M(a_1)||_F^2] +
        #          E_{a_4, a_5, a_6}[||M(a_6) T M(a_5) - M(a_4)||_F^2]) + lambda * ||T||_F^2
        # solution is:
        #     (E_{a_3}[M(a_3)^T M(a_3)] T E_{a_2}[M(a_2) M(a_2)^T] +
        #      E_{a_6}[M(a_6)^T M(a_6)] T E_{a_5}[M(a_5) M(a_5)^T]) + 2 * lambda * T =
        #     (E_{a_1, a_2, a_3}[M(a_3)^T M(a_1) M(a_2)^T] +
        #      E_{a_4, a_5, a_6}[M(a_6)^T M(a_4) M(a_5)^T])
        if solve_mode == "mean":
            arity_2_rtl_lhs_const = batch_kronecker_product(torch.bmm(mean_case[:, 2, :, :].transpose(1, 2), mean_case[:, 2, :, :]),
                                                            torch.bmm(mean_case[:, 1, :, :], mean_case[:, 1, :, :].transpose(1, 2))) + \
                                    batch_kronecker_product(torch.bmm(mean_case[:, 5, :, :].transpose(1, 2), mean_case[:, 5, :, :]),
                                                            torch.bmm(mean_case[:, 4, :, :], mean_case[:, 4, :, :].transpose(1, 2)))
        if solve_mode == "exact":
            arity_2_rtl_lhs_const = batch_kronecker_product(mean_left_square_case[:, 2, :, :], mean_right_square_case[:, 1, :, :]) + \
                                    batch_kronecker_product(mean_left_square_case[:, 5, :, :], mean_right_square_case[:, 4, :, :])
        arity_2_rtl_lhs_const += 2 * reg * torch.eye(
            matrix_size**2, device=obs.device).unsqueeze(0)
        arity_2_rtl_rhs_const = torch.bmm(torch.bmm(mean_case[:, 2, :, :].transpose(1, 2), mean_case[:, 0, :, :]),
                                          mean_case[:, 1, :, :].transpose(1, 2)) + \
                                torch.bmm(torch.bmm(mean_case[:, 5, :, :].transpose(1, 2), mean_case[:, 3, :, :]),
                                          mean_case[:, 4, :, :].transpose(1, 2))
        arity_2_rtl_rhs_const = arity_2_rtl_rhs_const.view(
            -1, matrix_size * matrix_size, 1)
        arity_2_rtl_ops = torch.solve(arity_2_rtl_rhs_const,
                                      arity_2_rtl_lhs_const)[0].view(
                                          -1, matrix_size, matrix_size)
        arity_2_rtl_inner_objs = 1.0 / 2 * (squared_norm(torch.bmm(torch.bmm(mean_case[:, 2, :, :], arity_2_rtl_ops),
                                                                   mean_case[:, 1, :, :]) - mean_case[:, 0, :, :], dim=(1, 2)) +
                                            squared_norm(torch.bmm(torch.bmm(mean_case[:, 5, :, :], arity_2_rtl_ops),
                                                                   mean_case[:, 4, :, :]) - mean_case[:, 3, :, :], dim=(1, 2))) + \
                                 reg * squared_norm(arity_2_rtl_ops, dim=(1, 2))

        # arity == 3
        # direction == "l"
        # inner obj is:
        #     1/5 (E_{a_1, a_2, a_4}[||T M(a_1, a_2) - M(a_4)||_F^2] +
        #          E_{a_2, a_3, a_5}[||T M(a_2, a_3) - M(a_5)||_F^2] +
        #          E_{a_3, a_1, a_6}[||T M(a_3, a_1) - M(a_6)||_F^2] +
        #          E_{a_4, a_5, a_7}[||T M(a_4, a_5) - M(a_7)||_F^2] +
        #          E_{a_5, a_6, a_8}[||T M(a_5, a_6) - M(a_8)||_F^2]) + lambda * ||T||_F^2
        # solution is:
        #     T (E_{a_1, a_2}[M(a_1, a_2) M(a_1, a_2)^T] +
        #        E_{a_2, a_3}[M(a_2, a_3) M(a_2, a_3)^T] +
        #        E_{a_3, a_1}[M(a_3, a_1) M(a_3, a_1)^T] +
        #        E_{a_4, a_5}[M(a_4, a_5) M(a_4, a_5)^T] +
        #        E_{a_5, a_6}[M(a_5, a_6) M(a_5, a_6)^T] + 5 * lambda * I) =
        #     (E_{a_1, a_2, a_4}[M(a_4) M(a_1, a_2)^T] +
        #      E_{a_2, a_3, a_5}[M(a_5) M(a_2, a_3)^T] +
        #      E_{a_3, a_1, a_6}[M(a_6) M(a_3, a_1)^T] +
        #      E_{a_4, a_5, a_7}[M(a_7) M(a_4, a_5)^T] +
        #      E_{a_5, a_6, a_8}[M(a_8) M(a_5, a_6)^T])
        # Note that E_{a_1, a_2}[[M(a_1); M(a_2)][M(a_1)^T, M(a_2)^T]]
        #           its i,j th block is E_{a_i, a_j}[M(a_i) M(a_j)^T]
        first_matrix = torch.cat(
            [mean_case[:, 0, :, :], mean_case[:, 1, :, :]], dim=1)
        second_matrix = torch.cat(
            [mean_case[:, 1, :, :], mean_case[:, 2, :, :]], dim=1)
        third_matrix = torch.cat(
            [mean_case[:, 2, :, :], mean_case[:, 0, :, :]], dim=1)
        fourth_matrix = torch.cat(
            [mean_case[:, 3, :, :], mean_case[:, 4, :, :]], dim=1)
        fifth_matrix = torch.cat(
            [mean_case[:, 4, :, :], mean_case[:, 5, :, :]], dim=1)
        sixth_matrix = torch.cat(
            [mean_case[:, 5, :, :], mean_case[:, 3, :, :]], dim=1)
        if solve_mode == "mean":
            common = torch.bmm(first_matrix, first_matrix.transpose(1, 2)) + \
                     torch.bmm(second_matrix, second_matrix.transpose(1, 2)) + \
                     torch.bmm(third_matrix, third_matrix.transpose(1, 2)) + \
                     torch.bmm(fourth_matrix, fourth_matrix.transpose(1, 2))
            arity_3_l_lhs_const = common + torch.bmm(
                fifth_matrix, fifth_matrix.transpose(1, 2))
        if solve_mode == "exact":
            comp_1 = torch.cat([
                torch.cat([
                    mean_right_square_case[:, 0, :, :],
                    torch.bmm(mean_case[:, 0, :, :],
                              mean_case[:, 1, :, :].transpose(1, 2))
                ],
                          dim=2),
                torch.cat([
                    torch.bmm(
                        mean_case[:, 1, :, :], mean_case[:, 0, :, :].transpose(
                            1, 2)), mean_right_square_case[:, 1, :, :]
                ],
                          dim=2)
            ],
                               dim=1)
            comp_2 = torch.cat([
                torch.cat([
                    mean_right_square_case[:, 1, :, :],
                    torch.bmm(mean_case[:, 1, :, :],
                              mean_case[:, 2, :, :].transpose(1, 2))
                ],
                          dim=2),
                torch.cat([
                    torch.bmm(
                        mean_case[:, 2, :, :], mean_case[:, 1, :, :].transpose(
                            1, 2)), mean_right_square_case[:, 2, :, :]
                ],
                          dim=2)
            ],
                               dim=1)
            comp_3 = torch.cat([
                torch.cat([
                    mean_right_square_case[:, 2, :, :],
                    torch.bmm(mean_case[:, 2, :, :],
                              mean_case[:, 0, :, :].transpose(1, 2))
                ],
                          dim=2),
                torch.cat([
                    torch.bmm(
                        mean_case[:, 0, :, :], mean_case[:, 2, :, :].transpose(
                            1, 2)), mean_right_square_case[:, 0, :, :]
                ],
                          dim=2)
            ],
                               dim=1)
            comp_4 = torch.cat([
                torch.cat([
                    mean_right_square_case[:, 3, :, :],
                    torch.bmm(mean_case[:, 3, :, :],
                              mean_case[:, 4, :, :].transpose(1, 2))
                ],
                          dim=2),
                torch.cat([
                    torch.bmm(
                        mean_case[:, 4, :, :], mean_case[:, 3, :, :].transpose(
                            1, 2)), mean_right_square_case[:, 4, :, :]
                ],
                          dim=2)
            ],
                               dim=1)
            common = comp_1 + comp_2 + comp_3 + comp_4
            arity_3_l_lhs_const = common + \
                                  torch.cat([torch.cat([mean_right_square_case[:, 4, :, :], torch.bmm(mean_case[:, 4, :, :], mean_case[:, 5, :, :].transpose(1, 2))], dim=2),
                                             torch.cat([torch.bmm(mean_case[:, 5, :, :], mean_case[:, 4, :, :].transpose(1, 2)), mean_right_square_case[:, 5, :, :]], dim=2)],
                                            dim=1)
        arity_3_l_lhs_const += 5 * reg * torch.eye(
            matrix_size * 2, device=obs.device).unsqueeze(0)
        arity_3_l_rhs_const = torch.bmm(mean_case[:, 3, :, :], first_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case[:, 4, :, :], second_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case[:, 5, :, :], third_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case[:, 6, :, :], fourth_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case[:, 7, :, :], fifth_matrix.transpose(1, 2))
        arity_3_l_ops = torch.solve(arity_3_l_rhs_const.transpose(1, 2),
                                    arity_3_l_lhs_const.transpose(
                                        1, 2))[0].transpose(1, 2)
        arity_3_l_inner_objs = 1.0 / 5 * (squared_norm(torch.bmm(arity_3_l_ops, first_matrix) - mean_case[:, 3, :, :], dim=(1, 2)) +
                                          squared_norm(torch.bmm(arity_3_l_ops, second_matrix) - mean_case[:, 4, :, :], dim=(1, 2)) +
                                          squared_norm(torch.bmm(arity_3_l_ops, third_matrix) - mean_case[:, 5, :, :], dim=(1, 2)) +
                                          squared_norm(torch.bmm(arity_3_l_ops, fourth_matrix) - mean_case[:, 6, :, :], dim=(1, 2)) +
                                          squared_norm(torch.bmm(arity_3_l_ops, fifth_matrix) - mean_case[:, 7, :, :], dim=(1, 2))) + \
                               reg * squared_norm(arity_3_l_ops, dim=(1, 2))

        # arity == 3
        # direction == "r"
        # inner obj is:
        #     1/5 (E_{a_1, a_2, a_5}[||T M(a_1, a_2) - M(a_5)||_F^2] +
        #          E_{a_2, a_3, a_6}[||T M(a_2, a_3) - M(a_6)||_F^2] +
        #          E_{a_3, a_1, a_4}[||T M(a_3, a_1) - M(a_4)||_F^2] +
        #          E_{a_4, a_5, a_8}[||T M(a_4, a_5) - M(a_8)||_F^2] +
        #          E_{a_6, a_4, a_7}[||T M(a_6, a_4) - M(a_7)||_F^2]) + lambda * ||T||_F^2
        # solution is:
        #     T (E_{a_1, a_2}[M(a_1, a_2) M(a_1, a_2)^T] +
        #        E_{a_2, a_3}[M(a_2, a_3) M(a_2, a_3)^T] +
        #        E_{a_3, a_1}[M(a_3, a_1) M(a_3, a_1)^T] +
        #        E_{a_4, a_5}[M(a_4, a_5) M(a_4, a_5)^T] +
        #        E_{a_6, a_4}[M(a_6, a_4) M(a_6, a_4)^T] + 5 * lambda * I) =
        #     (E_{a_1, a_2, a_5}[M(a_5) M(a_1, a_2)^T] +
        #      E_{a_2, a_3, a_6}[M(a_6) M(a_2, a_3)^T] +
        #      E_{a_3, a_1, a_4}[M(a_4) M(a_3, a_1)^T] +
        #      E_{a_4, a_5, a_8}[M(a_8) M(a_4, a_5)^T] +
        #      E_{a_6, a_4, a_7}[M(a_7) M(a_6, a_4)^T])
        # Note that E_{a_1, a_2}[[M(a_1); M(a_2)][M(a_1)^T, M(a_2)^T]]
        #           its i,j th block is E_{a_i, a_j}[M(a_i) M(a_j)^T]
        if solve_mode == "mean":
            arity_3_r_lhs_const = common + torch.bmm(
                sixth_matrix, sixth_matrix.transpose(1, 2))
        if solve_mode == "exact":
            arity_3_r_lhs_const = common + \
                                  torch.cat([torch.cat([mean_right_square_case[:, 5, :, :], torch.bmm(mean_case[:, 5, :, :], mean_case[:, 3, :, :].transpose(1, 2))], dim=2),
                                             torch.cat([torch.bmm(mean_case[:, 3, :, :], mean_case[:, 5, :, :].transpose(1, 2)), mean_right_square_case[:, 3, :, :]], dim=2)],
                                            dim=1)
        arity_3_r_lhs_const += 5 * reg * torch.eye(
            matrix_size * 2, device=obs.device).unsqueeze(0)
        arity_3_r_rhs_const = torch.bmm(mean_case[:, 4, :, :], first_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case[:, 5, :, :], second_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case[:, 3, :, :], third_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case[:, 7, :, :], fourth_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case[:, 6, :, :], sixth_matrix.transpose(1, 2))
        arity_3_r_ops = torch.solve(arity_3_r_rhs_const.transpose(1, 2),
                                    arity_3_r_lhs_const.transpose(
                                        1, 2))[0].transpose(1, 2)
        arity_3_r_inner_objs = 1.0 / 5 * (squared_norm(torch.bmm(arity_3_r_ops, first_matrix) - mean_case[:, 4, :, :], dim=(1, 2)) +
                                          squared_norm(torch.bmm(arity_3_r_ops, second_matrix) - mean_case[:, 5, :, :], dim=(1, 2)) +
                                          squared_norm(torch.bmm(arity_3_r_ops, third_matrix) - mean_case[:, 3, :, :], dim=(1, 2)) +
                                          squared_norm(torch.bmm(arity_3_r_ops, fourth_matrix) - mean_case[:, 7, :, :], dim=(1, 2)) +
                                          squared_norm(torch.bmm(arity_3_r_ops, sixth_matrix) - mean_case[:, 6, :, :], dim=(1, 2))) + \
                               reg * squared_norm(arity_3_r_ops, dim=(1, 2))

        return torch.stack([
            arity_1_inner_objs, arity_2_ltr_inner_objs, arity_2_rtl_inner_objs,
            arity_3_l_inner_objs, arity_3_r_inner_objs
        ],
                           dim=1), [
                               arity_1_ops, arity_2_ltr_ops, arity_2_rtl_ops,
                               arity_3_l_ops, arity_3_r_ops
                           ]

    def solve_logic(self, obs, space, reg, solve_mode="mean"):
        # obs: Tensor of shape (batch, 8, slot, case)
        # space: Tensor of shape (case, matrix_size, matrix_size)
        batch, _, slot, case = obs.shape
        _, matrix_size, _ = space.shape
        mean_case = torch.sum(
            obs.reshape(-1, case).unsqueeze(-1).unsqueeze(-1) *
            space.unsqueeze(0),
            dim=1)
        mean_case = mean_case.view(-1, 8, slot, matrix_size, matrix_size)
        mean_case_view = mean_case.transpose(1, 2).reshape(
            -1, 8, matrix_size, matrix_size)
        mean_case_slot_view = mean_case.view(-1, slot, matrix_size,
                                             matrix_size)

        # arity == 1
        # inner obj is:
        #     1/5 (E_pos[E_{a_1, a_2}[||T M(a_1^{neigh(pos)}) - M(a_2^pos)||_F^2]] +
        #          E_pos[E_{a_2, a_3}[||T M(a_2^{neigh(pos)}) - M(a_3^pos)||_F^2]] +
        #          E_pos[E_{a_4, a_5}[||T M(a_4^{neigh(pos)}) - M(a_5^pos)||_F^2]] +
        #          E_pos[E_{a_5, a_6}[||T M(a_5^{neigh(pos)}) - M(a_6^pos)||_F^2]] +
        #          E_pos[E_{a_7, a_8}[||T M(a_7^{neigh(pos)}) - M(a_8^pos)||_F^2]]) + lambda * ||T||_F^2
        # solution is:
        #     T(E_pos[E_{a_1}[M(a_1^{neigh(pos)}) M(a_1^{neigh(pos)})^T]] +
        #       E_pos[E_{a_2}[M(a_2^{neigh(pos)}) M(a_2^{neigh(pos)})^T]] +
        #       E_pos[E_{a_4}[M(a_4^{neigh(pos)}) M(a_4^{neigh(pos)})^T]] +
        #       E_pos[E_{a_5}[M(a_5^{neigh(pos)}) M(a_5^{neigh(pos)})^T]] +
        #       E_pos[E_{a_7}[M(a_7^{neigh(pos)}) M(a_7^{neigh(pos)})^T]] + 5 * lambda * I) =
        #     (E_pos[E_{a_1, a_2}[M(a_2^pos) M(a_1^{neigh(pos)})^T]] +
        #      E_pos[E_{a_2, a_3}[M(a_3^pos) M(a_2^{neigh(pos)})^T]] +
        #      E_pos[E_{a_4, a_5}[M(a_5^pos) M(a_4^{neigh(pos)})^T]] +
        #      E_pos[E_{a_5, a_6}[M(a_6^pos) M(a_5^{neigh(pos)})^T]] +
        #      E_pos[E_{a_7, a_8}[M(a_8^pos) M(a_7^{neigh(pos)})^T]])
        mean_case_neighbours = [
            torch.cat([
                mean_case[:, :, -1, :, :], mean_case[:, :, 0, :, :],
                mean_case[:, :, 1, :, :]
            ],
                      dim=2)
        ]
        for slot_idx in range(1, slot - 1):
            mean_case_neighbours.append(
                mean_case[:, :, slot_idx - 1:slot_idx + 2, :, :].view(
                    -1, 8, matrix_size * 3, matrix_size))
        mean_case_neighbours.append(
            torch.cat([
                mean_case[:, :, -2, :, :], mean_case[:, :, -1, :, :],
                mean_case[:, :, 0, :, :]
            ],
                      dim=2))
        mean_case_neighbours = torch.stack(mean_case_neighbours,
                                           dim=1).view(-1, 8, matrix_size * 3,
                                                       matrix_size)
        if solve_mode == "mean":
            arity_1_lhs_const = torch.bmm(mean_case_neighbours[:, 0, :, :], mean_case_neighbours[:, 0, :, :].transpose(1, 2)) + \
                                torch.bmm(mean_case_neighbours[:, 1, :, :], mean_case_neighbours[:, 1, :, :].transpose(1, 2)) + \
                                torch.bmm(mean_case_neighbours[:, 3, :, :], mean_case_neighbours[:, 3, :, :].transpose(1, 2)) + \
                                torch.bmm(mean_case_neighbours[:, 4, :, :], mean_case_neighbours[:, 4, :, :].transpose(1, 2)) + \
                                torch.bmm(mean_case_neighbours[:, 6, :, :], mean_case_neighbours[:, 6, :, :].transpose(1, 2))
            arity_1_lhs_const = torch.mean(arity_1_lhs_const.view(
                -1, slot, matrix_size * 3, matrix_size * 3),
                                           dim=1)
        if solve_mode == "exact":
            mean_right_square_case = torch.sum(
                obs.reshape(-1, case).unsqueeze(-1).unsqueeze(-1) *
                torch.bmm(space, space.transpose(1, 2)).unsqueeze(0),
                dim=1)
            mean_right_square_case = mean_right_square_case.view(
                -1, 8, slot, matrix_size, matrix_size)

            def build_matrix(slot_idx):
                pre_idx = slot_idx - 1
                _, post_idx = divmod(slot_idx + 1, slot)
                row_1 = torch.cat([
                    mean_right_square_case[:, :, pre_idx, :, :].view(
                        -1, matrix_size, matrix_size),
                    torch.bmm(
                        mean_case_slot_view[:, pre_idx, :, :],
                        mean_case_slot_view[:, slot_idx, :, :].transpose(1,
                                                                         2)),
                    torch.bmm(
                        mean_case_slot_view[:, pre_idx, :, :],
                        mean_case_slot_view[:, post_idx, :, :].transpose(1, 2))
                ],
                                  dim=-1)
                row_2 = torch.cat([
                    torch.bmm(
                        mean_case_slot_view[:, slot_idx, :, :],
                        mean_case_slot_view[:, pre_idx, :, :].transpose(1, 2)),
                    mean_right_square_case[:, :, slot_idx, :, :].view(
                        -1, matrix_size, matrix_size),
                    torch.bmm(
                        mean_case_slot_view[:, slot_idx, :, :],
                        mean_case_slot_view[:, post_idx, :, :].transpose(1, 2))
                ],
                                  dim=-1)
                row_3 = torch.cat([
                    torch.bmm(
                        mean_case_slot_view[:, post_idx, :, :],
                        mean_case_slot_view[:, pre_idx, :, :].transpose(1, 2)),
                    torch.bmm(
                        mean_case_slot_view[:, post_idx, :, :],
                        mean_case_slot_view[:, slot_idx, :, :].transpose(1,
                                                                         2)),
                    mean_right_square_case[:, :, post_idx, :, :].view(
                        -1, matrix_size, matrix_size)
                ],
                                  dim=-1)
                return torch.cat([row_1, row_2, row_3],
                                 dim=-2).view(-1, 8, matrix_size * 3,
                                              matrix_size * 3)

            arity_1_lhs_const_list = sum(
                [build_matrix(slot_idx) for slot_idx in range(slot)]) / slot
            arity_1_lhs_const = arity_1_lhs_const_list[:, 0, :, :] + \
                                arity_1_lhs_const_list[:, 1, :, :] + \
                                arity_1_lhs_const_list[:, 3, :, :] + \
                                arity_1_lhs_const_list[:, 4, :, :] + \
                                arity_1_lhs_const_list[:, 6, :, :]
        arity_1_lhs_const += 5 * reg * torch.eye(
            matrix_size * 3, device=obs.device).unsqueeze(0)
        arity_1_rhs_const = torch.bmm(mean_case_view[:, 1, :, :], mean_case_neighbours[:, 0, :, :].transpose(1, 2)) + \
                            torch.bmm(mean_case_view[:, 2, :, :], mean_case_neighbours[:, 1, :, :].transpose(1, 2)) + \
                            torch.bmm(mean_case_view[:, 4, :, :], mean_case_neighbours[:, 3, :, :].transpose(1, 2)) + \
                            torch.bmm(mean_case_view[:, 5, :, :], mean_case_neighbours[:, 4, :, :].transpose(1, 2)) + \
                            torch.bmm(mean_case_view[:, 7, :, :], mean_case_neighbours[:, 6, :, :].transpose(1, 2))
        arity_1_rhs_const = torch.mean(arity_1_rhs_const.view(
            -1, slot, matrix_size, matrix_size * 3),
                                       dim=1)
        arity_1_ops = torch.solve(arity_1_rhs_const.transpose(1, 2),
                                  arity_1_lhs_const.transpose(1,
                                                              2))[0].transpose(
                                                                  1, 2)
        arity_1_ops_expand = arity_1_ops.unsqueeze(1).expand(
            -1, slot, -1, -1).reshape(-1, matrix_size, matrix_size * 3)
        arity_1_inner_objs = squared_norm(torch.bmm(arity_1_ops_expand, mean_case_neighbours[:, 0, :, :]) -
                                          mean_case_view[:, 1, :, :], dim=(1, 2)) + \
                             squared_norm(torch.bmm(arity_1_ops_expand, mean_case_neighbours[:, 1, :, :]) -
                                          mean_case_view[:, 2, :, :], dim=(1, 2)) + \
                             squared_norm(torch.bmm(arity_1_ops_expand, mean_case_neighbours[:, 3, :, :]) -
                                          mean_case_view[:, 4, :, :], dim=(1, 2)) + \
                             squared_norm(torch.bmm(arity_1_ops_expand, mean_case_neighbours[:, 4, :, :]) -
                                          mean_case_view[:, 5, :, :], dim=(1, 2)) + \
                             squared_norm(torch.bmm(arity_1_ops_expand, mean_case_neighbours[:, 6, :, :]) -
                                          mean_case_view[:, 7, :, :], dim=(1, 2))
        arity_1_inner_objs = 1.0 / 5 * torch.mean(
            arity_1_inner_objs.view(-1, slot), dim=1) + reg * squared_norm(
                arity_1_ops, dim=(1, 2))

        # arity == 2
        # direction: left to right
        # inner obj is:
        #     1/2 (E_pos[E_{a_1, a_2, a_3}[||T V(a_1^pos) \otimes V(a_2^pos) - V(a_3^pos)||_F^2]] +
        #          E_pos[E_{a_4, a_5, a_6}[||T V(a_4^pos) \otimes V(a_5^pos) - V(a_6^pos)||_F^2]]) + lambda * ||T||_F^2
        # solution is:
        #     T (E_pos[E_{a_1}[V(a_1^pos) V(a_1^pos)^T] \otimes E_{a_2}[V(a_2^pos) V(a_2^pos)^T]] +
        #        E_pos[E_{a_4}[V(a_4^pos) V(a_4^pos)^T] \otimes E_{a_5}[V(a_5^pos) V(a_5^pos)^T]] + 2 * lambda * I) =
        #     (E_pos[E_{a_1, a_2, a_3}[V(a_3^pos) (V(a_1^pos) \otimes V(a_2^pos))^T]] +
        #      E_pos[E_{a_4, a_5, a_6}[V(a_6^pos) (V(a_4^pos) \otimes V(a_5^pos))^T]])
        # Note that the objective is per-element bilinear
        # Kronecker product property: https://en.wikipedia.org/wiki/Kronecker_product
        if solve_mode == "mean":
            arity_2_ltr_lhs_const = batch_kronecker_product(torch.bmm(mean_case_view[:, 0, :, :].view(-1, matrix_size ** 2, 1),
                                                                      mean_case_view[:, 0, :, :].view(-1, 1, matrix_size ** 2)),
                                                            torch.bmm(mean_case_view[:, 1, :, :].view(-1, matrix_size ** 2, 1),
                                                                      mean_case_view[:, 1, :, :].view(-1, 1, matrix_size ** 2))) + \
                                    batch_kronecker_product(torch.bmm(mean_case_view[:, 3, :, :].view(-1, matrix_size ** 2, 1),
                                                                      mean_case_view[:, 3, :, :].view(-1, 1, matrix_size ** 2)),
                                                            torch.bmm(mean_case_view[:, 4, :, :].view(-1, matrix_size ** 2, 1),
                                                                      mean_case_view[:, 4, :, :].view(-1, 1, matrix_size ** 2)))
        if solve_mode == "exact":
            vec_square = torch.bmm(space.view(-1, matrix_size**2, 1),
                                   space.view(-1, 1,
                                              matrix_size**2)).unsqueeze(0)
            mean_vec_square_case_0 = torch.sum(
                obs[:, 0, :, :].reshape(-1, case).unsqueeze(-1).unsqueeze(-1) *
                vec_square,
                dim=1)
            mean_vec_square_case_1 = torch.sum(
                obs[:, 1, :, :].reshape(-1, case).unsqueeze(-1).unsqueeze(-1) *
                vec_square,
                dim=1)
            mean_vec_square_case_3 = torch.sum(
                obs[:, 3, :, :].reshape(-1, case).unsqueeze(-1).unsqueeze(-1) *
                vec_square,
                dim=1)
            mean_vec_square_case_4 = torch.sum(
                obs[:, 4, :, :].reshape(-1, case).unsqueeze(-1).unsqueeze(-1) *
                vec_square,
                dim=1)
            arity_2_ltr_lhs_const = batch_kronecker_product(mean_vec_square_case_0, mean_vec_square_case_1) + \
                                    batch_kronecker_product(mean_vec_square_case_3, mean_vec_square_case_4)
        arity_2_ltr_lhs_const = torch.mean(arity_2_ltr_lhs_const.view(
            -1, slot, matrix_size**4, matrix_size**4),
                                           dim=1)
        arity_2_ltr_lhs_const += 2 * (reg * 1000) * torch.eye(
            matrix_size**4, device=obs.device).unsqueeze(0)
        arity_2_ltr_rhs_const = torch.bmm(mean_case_view[:, 2, :, :].view(-1, matrix_size ** 2, 1),
                                          batch_kronecker_product(mean_case_view[:, 0, :, :].view(-1, 1, matrix_size ** 2),
                                                                  mean_case_view[:, 1, :, :].view(-1, 1, matrix_size ** 2))) + \
                                torch.bmm(mean_case_view[:, 5, :, :].view(-1, matrix_size ** 2, 1),
                                          batch_kronecker_product(mean_case_view[:, 3, :, :].view(-1, 1, matrix_size ** 2),
                                                                  mean_case_view[:, 4, :, :].view(-1, 1, matrix_size ** 2)))
        arity_2_ltr_rhs_const = torch.mean(arity_2_ltr_rhs_const.view(
            -1, slot, matrix_size**2, matrix_size**4),
                                           dim=1)
        arity_2_ltr_ops = torch.solve(arity_2_ltr_rhs_const.transpose(1, 2),
                                      arity_2_ltr_lhs_const.transpose(
                                          1, 2))[0].transpose(1, 2)
        arity_2_ltr_ops_expand = arity_2_ltr_ops.unsqueeze(1).expand(
            -1, slot, -1, -1).reshape(-1, matrix_size**2, matrix_size**4)
        arity_2_ltr_inner_objs = squared_norm(torch.bmm(arity_2_ltr_ops_expand, batch_kronecker_product(mean_case_view[:, 0, :, :].view(-1, matrix_size ** 2, 1),
                                                                                                        mean_case_view[:, 1, :, :].view(-1, matrix_size ** 2, 1))) -
                                              mean_case_view[:, 2, :, :].view(-1, matrix_size ** 2, 1), dim=(1, 2)) + \
                                 squared_norm(torch.bmm(arity_2_ltr_ops_expand, batch_kronecker_product(mean_case_view[:, 3, :, :].view(-1, matrix_size ** 2, 1),
                                                                                                        mean_case_view[:, 4, :, :].view(-1, matrix_size ** 2, 1))) -
                                              mean_case_view[:, 5, :, :].view(-1, matrix_size ** 2, 1), dim=(1, 2))
        arity_2_ltr_inner_objs = 1.0 / 2 * torch.mean(
            arity_2_ltr_inner_objs.view(-1, slot),
            dim=1) + (reg * 1000) * squared_norm(arity_2_ltr_ops, dim=(1, 2))

        # arity == 3
        # direction == "l"
        # inner obj is:
        #     1/5 (E_pos[E_{a_1, a_2, a_4}[||T M(a_1^pos, a_2^pos) - M(a_4^pos)||_F^2]] +
        #          E_pos[E_{a_2, a_3, a_5}[||T M(a_2^pos, a_3^pos) - M(a_5^pos)||_F^2]] +
        #          E_pos[E_{a_3, a_1, a_6}[||T M(a_3^pos, a_1^pos) - M(a_6^pos)||_F^2]] +
        #          E_pos[E_{a_4, a_5, a_7}[||T M(a_4^pos, a_5^pos) - M(a_7^pos)||_F^2]] +
        #          E_pos[E_{a_5, a_6, a_8}[||T M(a_5^pos, a_6^pos) - M(a_8^pos)||_F^2]]) + lambda * ||T||_F^2
        # solution is:
        #     T (E_pos[E_{a_1, a_2}[M(a_1^pos, a_2^pos) M(a_1^pos, a_2^pos)^T]] +
        #        E_pos[E_{a_2, a_3}[M(a_2^pos, a_3^pos) M(a_2^pos, a_3^pos)^T]] +
        #        E_pos[E_{a_3, a_1}[M(a_3^pos, a_1^pos) M(a_3^pos, a_1^pos)^T]] +
        #        E_pos[E_{a_4, a_5}[M(a_4^pos, a_5^pos) M(a_4^pos, a_5^pos)^T]] +
        #        E_pos[E_{a_5, a_6}[M(a_5^pos, a_6^pos) M(a_5^pos, a_6^pos)^T]] + 5 * lambda * I) =
        #     (E_pos[E_{a_1, a_2, a_4}[M(a_4^pos) M(a_1^pos, a_2^pos)^T]] +
        #      E_pos[E_{a_2, a_3, a_5}[M(a_5^pos) M(a_2^pos, a_3^pos)^T]] +
        #      E_pos[E_{a_3, a_1, a_6}[M(a_6^pos) M(a_3^pos, a_1^pos)^T]] +
        #      E_pos[E_{a_4, a_5, a_7}[M(a_7^pos) M(a_4^pos, a_5^pos)^T]] +
        #      E_pos[E_{a_5, a_6, a_8}[M(a_8^pos) M(a_5^pos, a_6^pos)^T]])
        first_matrix = torch.cat(
            [mean_case_view[:, 0, :, :], mean_case_view[:, 1, :, :]], dim=1)
        second_matrix = torch.cat(
            [mean_case_view[:, 1, :, :], mean_case_view[:, 2, :, :]], dim=1)
        third_matrix = torch.cat(
            [mean_case_view[:, 2, :, :], mean_case_view[:, 0, :, :]], dim=1)
        fourth_matrix = torch.cat(
            [mean_case_view[:, 3, :, :], mean_case_view[:, 4, :, :]], dim=1)
        fifth_matrix = torch.cat(
            [mean_case_view[:, 4, :, :], mean_case_view[:, 5, :, :]], dim=1)
        sixth_matrix = torch.cat(
            [mean_case_view[:, 5, :, :], mean_case_view[:, 3, :, :]], dim=1)
        if solve_mode == "mean":
            common = torch.bmm(first_matrix, first_matrix.transpose(1, 2)) + \
                     torch.bmm(second_matrix, second_matrix.transpose(1, 2)) + \
                     torch.bmm(third_matrix, third_matrix.transpose(1, 2)) + \
                     torch.bmm(fourth_matrix, fourth_matrix.transpose(1, 2))
            arity_3_l_lhs_const = common + torch.bmm(
                fifth_matrix, fifth_matrix.transpose(1, 2))
        if solve_mode == "exact":
            comp_1 = torch.cat([
                torch.cat([
                    mean_right_square_case[:, 0, :, :, :].reshape(
                        -1, matrix_size, matrix_size),
                    torch.bmm(mean_case_view[:, 0, :, :],
                              mean_case_view[:, 1, :, :].transpose(1, 2))
                ],
                          dim=2),
                torch.cat([
                    torch.bmm(mean_case_view[:, 1, :, :],
                              mean_case_view[:, 0, :, :].transpose(1, 2)),
                    mean_right_square_case[:, 1, :, :, :].reshape(
                        -1, matrix_size, matrix_size)
                ],
                          dim=2)
            ],
                               dim=1)
            comp_2 = torch.cat([
                torch.cat([
                    mean_right_square_case[:, 1, :, :, :].reshape(
                        -1, matrix_size, matrix_size),
                    torch.bmm(mean_case_view[:, 1, :, :],
                              mean_case_view[:, 2, :, :].transpose(1, 2))
                ],
                          dim=2),
                torch.cat([
                    torch.bmm(mean_case_view[:, 2, :, :],
                              mean_case_view[:, 1, :, :].transpose(1, 2)),
                    mean_right_square_case[:, 2, :, :, :].reshape(
                        -1, matrix_size, matrix_size)
                ],
                          dim=2)
            ],
                               dim=1)
            comp_3 = torch.cat([
                torch.cat([
                    mean_right_square_case[:, 2, :, :, :].reshape(
                        -1, matrix_size, matrix_size),
                    torch.bmm(mean_case_view[:, 2, :, :],
                              mean_case_view[:, 0, :, :].transpose(1, 2))
                ],
                          dim=2),
                torch.cat([
                    torch.bmm(mean_case_view[:, 0, :, :],
                              mean_case_view[:, 2, :, :].transpose(1, 2)),
                    mean_right_square_case[:, 0, :, :, :].reshape(
                        -1, matrix_size, matrix_size)
                ],
                          dim=2)
            ],
                               dim=1)
            comp_4 = torch.cat([
                torch.cat([
                    mean_right_square_case[:, 3, :, :, :].reshape(
                        -1, matrix_size, matrix_size),
                    torch.bmm(mean_case_view[:, 3, :, :],
                              mean_case_view[:, 4, :, :].transpose(1, 2))
                ],
                          dim=2),
                torch.cat([
                    torch.bmm(mean_case_view[:, 4, :, :],
                              mean_case_view[:, 3, :, :].transpose(1, 2)),
                    mean_right_square_case[:, 4, :, :, :].reshape(
                        -1, matrix_size, matrix_size)
                ],
                          dim=2)
            ],
                               dim=1)
            common = comp_1 + comp_2 + comp_3 + comp_4
            arity_3_l_lhs_const = common + \
                                  torch.cat([torch.cat([mean_right_square_case[:, 4, :, :, :].reshape(-1, matrix_size, matrix_size),
                                                       torch.bmm(mean_case_view[:, 4, :, :], mean_case_view[:, 5, :, :].transpose(1, 2))], dim=2),
                                             torch.cat([torch.bmm(mean_case_view[:, 5, :, :], mean_case_view[:, 4, :, :].transpose(1, 2)),
                                                        mean_right_square_case[:, 5, :, :, :].reshape(-1, matrix_size, matrix_size)], dim=2)], dim=1)
        arity_3_l_lhs_const = torch.mean(arity_3_l_lhs_const.view(
            -1, slot, matrix_size * 2, matrix_size * 2),
                                         dim=1)
        arity_3_l_lhs_const += 5 * reg * torch.eye(
            matrix_size * 2, device=obs.device).unsqueeze(0)
        arity_3_l_rhs_const = torch.bmm(mean_case_view[:, 3, :, :], first_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case_view[:, 4, :, :], second_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case_view[:, 5, :, :], third_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case_view[:, 6, :, :], fourth_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case_view[:, 7, :, :], fifth_matrix.transpose(1, 2))
        arity_3_l_rhs_const = torch.mean(arity_3_l_rhs_const.view(
            -1, slot, matrix_size, matrix_size * 2),
                                         dim=1)
        arity_3_l_ops = torch.solve(arity_3_l_rhs_const.transpose(1, 2),
                                    arity_3_l_lhs_const.transpose(
                                        1, 2))[0].transpose(1, 2)
        arity_3_l_ops_expand = arity_3_l_ops.unsqueeze(1).expand(
            -1, slot, -1, -1).reshape(-1, matrix_size, matrix_size * 2)
        arity_3_l_inner_objs = squared_norm(torch.bmm(arity_3_l_ops_expand, first_matrix) - mean_case_view[:, 3, :, :], dim=(1, 2)) + \
                               squared_norm(torch.bmm(arity_3_l_ops_expand, second_matrix) - mean_case_view[:, 4, :, :], dim=(1, 2)) + \
                               squared_norm(torch.bmm(arity_3_l_ops_expand, third_matrix) - mean_case_view[:, 5, :, :], dim=(1, 2)) + \
                               squared_norm(torch.bmm(arity_3_l_ops_expand, fourth_matrix) - mean_case_view[:, 6, :, :], dim=(1, 2)) + \
                               squared_norm(torch.bmm(arity_3_l_ops_expand, fifth_matrix) - mean_case_view[:, 7, :, :], dim=(1, 2))
        arity_3_l_inner_objs = 1.0 / 5 * torch.mean(
            arity_3_l_inner_objs.view(-1, slot), dim=1) + reg * squared_norm(
                arity_3_l_ops, dim=(1, 2))

        # arity == 3
        # direction == "r"
        # inner obj is:
        #     1/5 (E_pos[E_{a_1, a_2, a_5}[||T M(a_1^pos, a_2^pos) - M(a_5^pos)||_F^2]] +
        #          E_pos[E_{a_2, a_3, a_6}[||T M(a_2^pos, a_3^pos) - M(a_6^pos)||_F^2]] +
        #          E_pos[E_{a_3, a_1, a_4}[||T M(a_3^pos, a_1^pos) - M(a_4^pos)||_F^2]] +
        #          E_pos[E_{a_4, a_5, a_8}[||T M(a_4^pos, a_5^pos) - M(a_8^pos)||_F^2]] +
        #          E_pos[E_{a_6, a_4, a_7}[||T M(a_6^pos, a_4^pos) - M(a_7^pos)||_F^2]]) + lambda * ||T||_F^2
        # solution is:
        #     T (E_pos[E_{a_1, a_2}[M(a_1^pos, a_2^pos) M(a_1^pos, a_2^pos)^T]] +
        #        E_pos[E_{a_2, a_3}[M(a_2^pos, a_3^pos) M(a_2^pos, a_3^pos)^T]] +
        #        E_pos[E_{a_3, a_1}[M(a_3^pos, a_1^pos) M(a_3^pos, a_1^pos)^T]] +
        #        E_pos[E_{a_4, a_5}[M(a_4^pos, a_5^pos) M(a_4^pos, a_5^pos)^T]] +
        #        E_pos[E_{a_6, a_4}[M(a_6^pos, a_4^pos) M(a_6^pos, a_4^pos)^T]] + 5 * lambda * I) =
        #     (E_pos[E_{a_1, a_2, a_5}[M(a_5^pos) M(a_1^pos, a_2^pos)^T]] +
        #      E_pos[E_{a_2, a_3, a_6}[M(a_6^pos) M(a_2^pos, a_3^pos)^T]] +
        #      E_pos[E_{a_3, a_1, a_4}[M(a_4^pos) M(a_3^pos, a_1^pos)^T]] +
        #      E_pos[E_{a_4, a_5, a_8}[M(a_8^pos) M(a_4^pos, a_5^pos)^T]] +
        #      E_pos[E_{a_6, a_4, a_7}[M(a_7^pos) M(a_6^pos, a_4^pos)^T]])
        if solve_mode == "mean":
            arity_3_r_lhs_const = common + torch.bmm(
                sixth_matrix, sixth_matrix.transpose(1, 2))
        if solve_mode == "exact":
            arity_3_r_lhs_const = common + \
                                  torch.cat([torch.cat([mean_right_square_case[:, 5, :, :, :].reshape(-1, matrix_size, matrix_size),
                                                       torch.bmm(mean_case_view[:, 5, :, :], mean_case_view[:, 3, :, :].transpose(1, 2))], dim=2),
                                             torch.cat([torch.bmm(mean_case_view[:, 3, :, :], mean_case_view[:, 5, :, :].transpose(1, 2)),
                                                        mean_right_square_case[:, 3, :, :, :].reshape(-1, matrix_size, matrix_size)], dim=2)], dim=1)
        arity_3_r_lhs_const = torch.mean(arity_3_r_lhs_const.view(
            -1, slot, matrix_size * 2, matrix_size * 2),
                                         dim=1)
        arity_3_r_lhs_const += 5 * reg * torch.eye(
            matrix_size * 2, device=obs.device).unsqueeze(0)
        arity_3_r_rhs_const = torch.bmm(mean_case_view[:, 4, :, :], first_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case_view[:, 5, :, :], second_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case_view[:, 3, :, :], third_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case_view[:, 7, :, :], fourth_matrix.transpose(1, 2)) + \
                              torch.bmm(mean_case_view[:, 6, :, :], sixth_matrix.transpose(1, 2))
        arity_3_r_rhs_const = torch.mean(arity_3_r_rhs_const.view(
            -1, slot, matrix_size, matrix_size * 2),
                                         dim=1)
        arity_3_r_ops = torch.solve(arity_3_r_rhs_const.transpose(1, 2),
                                    arity_3_r_lhs_const.transpose(
                                        1, 2))[0].transpose(1, 2)
        arity_3_r_ops_expand = arity_3_r_ops.unsqueeze(1).expand(
            -1, slot, -1, -1).reshape(-1, matrix_size, matrix_size * 2)
        arity_3_r_inner_objs = squared_norm(torch.bmm(arity_3_r_ops_expand, first_matrix) - mean_case_view[:, 4, :, :], dim=(1, 2)) + \
                               squared_norm(torch.bmm(arity_3_r_ops_expand, second_matrix) - mean_case_view[:, 5, :, :], dim=(1, 2)) + \
                               squared_norm(torch.bmm(arity_3_r_ops_expand, third_matrix) - mean_case_view[:, 3, :, :], dim=(1, 2)) + \
                               squared_norm(torch.bmm(arity_3_r_ops_expand, fourth_matrix) - mean_case_view[:, 7, :, :], dim=(1, 2)) + \
                               squared_norm(torch.bmm(arity_3_r_ops_expand, sixth_matrix) - mean_case_view[:, 6, :, :], dim=(1, 2))
        arity_3_r_inner_objs = 1.0 / 5 * torch.mean(
            arity_3_r_inner_objs.view(-1, slot), dim=1) + reg * squared_norm(
                arity_3_r_ops, dim=(1, 2))

        return torch.stack([arity_1_inner_objs, arity_2_ltr_inner_objs, arity_3_l_inner_objs, arity_3_r_inner_objs], dim=1), \
               [arity_1_ops, arity_2_ltr_ops, arity_3_l_ops, arity_3_r_ops]
