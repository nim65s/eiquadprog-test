//
// Copyright (c) 2019-2020 CNRS
//
// This file has been imported from eiquadprog and stripped from Boost.
//
// eiquadprog is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
//(at your option) any later version.

// eiquadprog is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public License
// along with eiquadprog.  If not, see <https://www.gnu.org/licenses/>.

#include <iostream>

#include <Eigen/Core>

#include "eiquadprog/eiquadprog.hpp"

// The problem is in the form:
// min 0.5 * x G x + g0 x
// s.t.
// CE^T x + ce0 = 0
// CI^T x + ci0 >= 0
// The matrix and vectors dimensions are as follows:
// G: n * n
// g0: n
// CE: n * p
// ce0: p
// CI: n * m
// ci0: m
// x: n

// min ||x||^2

int test_unbiased() {
    int ret = 0;
    Eigen::MatrixXd Q(2, 2);
    Q.setZero();
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;

    Eigen::VectorXd C(2);
    C.setZero();

    Eigen::MatrixXd Aeq(2, 0);

    Eigen::VectorXd Beq(0);

    Eigen::MatrixXd Aineq(2, 0);

    Eigen::VectorXd Bineq(0);

    Eigen::VectorXd x(2);
    Eigen::VectorXi activeSet(0);
    size_t activeSetSize;

    Eigen::VectorXd solution(2);
    solution.setZero();

    double val = 0.0;

    double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

    if (fabs(out - val) > 1e-6) ret++;
    if (!x.isApprox(solution)) ret++;
    return ret;
}

// min ||x-x_0||^2, x_0 = (1 1)^T

int test_biased() {
    int ret=0;
    Eigen::MatrixXd Q(2, 2);
    Q.setZero();
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;

    Eigen::VectorXd C(2);
    C(0) = -1.;
    C(1) = -1.;

    Eigen::MatrixXd Aeq(2, 0);

    Eigen::VectorXd Beq(0);

    Eigen::MatrixXd Aineq(2, 0);

    Eigen::VectorXd Bineq(0);

    Eigen::VectorXd x(2);
    Eigen::VectorXi activeSet(0);
    size_t activeSetSize;

    Eigen::VectorXd solution(2);
    solution(0) = 1.;
    solution(1) = 1.;

    double val = -1.;

    double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

    if (fabs(out - val) > 1e-6) ret++;
    if (!x.isApprox(solution)) ret++;
    return ret;
}

// min ||x||^2
//    s.t.
// x[1] = 1 - x[0]

int test_equality_constraints() {
    int ret=0;
    Eigen::MatrixXd Q(2, 2);
    Q.setZero();
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;

    Eigen::VectorXd C(2);
    C.setZero();

    Eigen::MatrixXd Aeq(2, 1);
    Aeq(0, 0) = 1.;
    Aeq(1, 0) = 1.;

    Eigen::VectorXd Beq(1);
    Beq(0) = -1.;

    Eigen::MatrixXd Aineq(2, 0);

    Eigen::VectorXd Bineq(0);

    Eigen::VectorXd x(2);
    Eigen::VectorXi activeSet(1);
    size_t activeSetSize;

    Eigen::VectorXd solution(2);
    solution(0) = 0.5;
    solution(1) = 0.5;

    double val = 0.25;

    double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

    if (fabs(out - val) > 1e-6) ret++;
    if (!x.isApprox(solution)) ret++;
    return ret;
}

// min ||x||^2
//    s.t.
// x[i] >= 1

int test_inequality_constraints() {
    int ret = 0;
    Eigen::MatrixXd Q(2, 2);
    Q.setZero();
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;

    Eigen::VectorXd C(2);
    C.setZero();

    Eigen::MatrixXd Aeq(2, 0);

    Eigen::VectorXd Beq(0);

    Eigen::MatrixXd Aineq(2, 2);
    Aineq.setZero();
    Aineq(0, 0) = 1.;
    Aineq(1, 1) = 1.;

    Eigen::VectorXd Bineq(2);
    Bineq(0) = -1.;
    Bineq(1) = -1.;

    Eigen::VectorXd x(2);
    Eigen::VectorXi activeSet(2);
    size_t activeSetSize;

    Eigen::VectorXd solution(2);
    solution(0) = 1.;
    solution(1) = 1.;

    double val = 1.;

    double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

    if (fabs(out - val) > 1e-6) ret++;
    if (!x.isApprox(solution)) ret++;
    return ret;
}

// min ||x-x_0||^2, x_0 = (1 1)^T
//    s.t.
// x[1] = 5 - x[0]
// x[1] >= 3

int test_full() {
    int ret=0;
    Eigen::MatrixXd Q(2, 2);
    Q.setZero();
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;

    Eigen::VectorXd C(2);
    C(0) = -1.;
    C(1) = -1.;

    Eigen::MatrixXd Aeq(2, 1);
    Aeq(0, 0) = 1.;
    Aeq(1, 0) = 1.;

    Eigen::VectorXd Beq(1);
    Beq(0) = -5.;

    Eigen::MatrixXd Aineq(2, 1);
    Aineq.setZero();
    Aineq(1, 0) = 1.;

    Eigen::VectorXd Bineq(1);
    Bineq(0) = -3.;

    Eigen::VectorXd x(2);
    Eigen::VectorXi activeSet(2);
    size_t activeSetSize;

    Eigen::VectorXd solution(2);
    solution(0) = 2.;
    solution(1) = 3.;

    double val = 1.5;

    double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

    if (fabs(out - val) > 1e-6) ret++;
    if (!x.isApprox(solution)) ret++;
    return ret;
}

// min ||x||^2
//    s.t.
// x[0] =  1
// x[0] = -1
// DOES NOT WORK!

int test_unfeasible_equalities() {
    int ret=0;
    Eigen::MatrixXd Q(2, 2);
    Q.setZero();
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;

    Eigen::VectorXd C(2);
    C.setZero();

    Eigen::MatrixXd Aeq(2, 2);
    Aeq.setZero();
    Aeq(0, 0) = 1.;
    Aeq(0, 1) = 1.;

    Eigen::VectorXd Beq(2);
    Beq(0) = -1.;
    Beq(1) = 1.;

    Eigen::MatrixXd Aineq(2, 0);

    Eigen::VectorXd Bineq(0);

    Eigen::VectorXd x(2);
    Eigen::VectorXi activeSet(2);
    size_t activeSetSize;

    double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

    // DOES NOT WORK!?
    if (!std::isinf(out)) ret++;
    return ret;
}

// min ||x||^2
//    s.t.
// x[0] >=  1
// x[0] <= -1

int test_unfeasible_inequalities() {
    int ret =0;
    Eigen::MatrixXd Q(2, 2);
    Q.setZero();
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;

    Eigen::VectorXd C(2);
    C.setZero();

    Eigen::MatrixXd Aeq(2, 0);

    Eigen::VectorXd Beq(0);

    Eigen::MatrixXd Aineq(2, 2);
    Aineq.setZero();
    Aineq(0, 0) = 1.;
    Aineq(0, 1) = -1.;

    Eigen::VectorXd Bineq(2);
    Bineq(0) = -1;
    Bineq(1) = -1;

    Eigen::VectorXd x(2);
    Eigen::VectorXi activeSet(2);
    size_t activeSetSize;

    double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

    if (!std::isinf(out)) ret++;
    return ret;
}

// min ||x-x_0||^2, x_0 = (1 1)^T
//    s.t.
// x[1] = 1 - x[0]
// x[0] <= 0
// x[1] <= 0

int test_unfeasible_constraints() {
    int ret=0;
    Eigen::MatrixXd Q(2, 2);
    Q.setZero();
    Q(0, 0) = 1.0;
    Q(1, 1) = 1.0;

    Eigen::VectorXd C(2);
    C(0) = -1.;
    C(1) = -1.;

    Eigen::MatrixXd Aeq(2, 1);
    Aeq(0, 0) = 1.;
    Aeq(1, 0) = 1.;

    Eigen::VectorXd Beq(1);
    Beq(0) = -1.;

    Eigen::MatrixXd Aineq(2, 2);
    Aineq.setZero();
    Aineq(0, 0) = -1.;
    Aineq(1, 1) = -1.;

    Eigen::VectorXd Bineq(2);
    Bineq.setZero();

    Eigen::VectorXd x(2);
    Eigen::VectorXi activeSet(3);
    size_t activeSetSize;

    double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

    if (!std::isinf(out)) ret++;
    return ret;
}

// min -||x||^2
// DOES NOT WORK!

int test_unbounded() {
    int ret=0;
    Eigen::MatrixXd Q(2, 2);
    Q.setZero();
    Q(0, 0) = -1.0;
    Q(1, 1) = -1.0;

    Eigen::VectorXd C(2);
    C.setZero();

    Eigen::MatrixXd Aeq(2, 0);

    Eigen::VectorXd Beq(0);

    Eigen::MatrixXd Aineq(2, 0);

    Eigen::VectorXd Bineq(0);

    Eigen::VectorXd x(2);
    Eigen::VectorXi activeSet(0);
    size_t activeSetSize;

    double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

    // DOES NOT WORK!?
    if (!std::isinf(out)) ret++;
    return ret;
}

// min -||x||^2
//    s.t.
// 0<= x[0] <= 1
// 0<= x[1] <= 1
// DOES NOT WORK!

int test_nonconvex() {
    int ret=0;
    Eigen::MatrixXd Q(2, 2);
    Q.setZero();
    Q(0, 0) = -1.0;
    Q(1, 1) = -1.0;

    Eigen::VectorXd C(2);
    C.setZero();

    Eigen::MatrixXd Aeq(2, 0);

    Eigen::VectorXd Beq(0);

    Eigen::MatrixXd Aineq(2, 4);
    Aineq.setZero();
    Aineq(0, 0) = 1.;
    Aineq(0, 1) = -1.;
    Aineq(1, 2) = 1.;
    Aineq(1, 3) = -1.;

    Eigen::VectorXd Bineq(4);
    Bineq(0) = 0.;
    Bineq(1) = 1.;
    Bineq(2) = 0.;
    Bineq(3) = 1.;

    Eigen::VectorXd x(2);
    Eigen::VectorXi activeSet(4);
    size_t activeSetSize;

    Eigen::VectorXd solution(2);
    solution(0) = 1.;
    solution(1) = 1.;

    double val = -1.;

    double out = Eigen::solve_quadprog(Q, C, Aeq, Beq, Aineq, Bineq, x, activeSet, activeSetSize);

    // DOES NOT WORK!?
    if (fabs(out - val) > 1e-6) ret++;
    if (!x.isApprox(solution)) ret++;
    return ret;
}


int main() {
    int unbiased = test_unbiased();
    int biased = test_biased();
    int equality_constraints = test_equality_constraints();
    int inequality_constraints = test_inequality_constraints();
    int full = test_full();
    int unfeasible_equalities = test_unfeasible_equalities();
    int unfeasible_inequalities = test_unfeasible_inequalities();
    int unfeasible_constraints = test_unfeasible_constraints();
    int unbounded = test_unbounded();
    int nonconvex = test_nonconvex();

    std::cout << "unbiased: 0/" << unbiased << std::endl;
    std::cout << "biased: 0/" << biased << std::endl;
    std::cout << "equality_constraints: 0/" << equality_constraints << std::endl;
    std::cout << "inequality_constraints: 0/" << inequality_constraints << std::endl;
    std::cout << "full: 0/" << full << std::endl;
    std::cout << "unfeasible_equalities: 1/" << unfeasible_equalities << std::endl;
    std::cout << "unfeasible_inequalities: 0/" << unfeasible_inequalities << std::endl;
    std::cout << "unfeasible_constraints: 0/" << unfeasible_constraints << std::endl;
    std::cout << "unbounded: 1/" << unbounded << std::endl;
    std::cout << "nonconvex: 2/" << nonconvex << std::endl;

    return unbiased + biased + equality_constraints + inequality_constraints + full + unfeasible_inequalities + unfeasible_constraints;
}
