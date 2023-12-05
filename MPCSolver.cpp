/*
 * @Description:
 * @Version: 2.0
 * @Author: ZHAO B.T.
 * @Date: 2023-11-23 12:01:36
 * @LastEditors: ZHAO B.T.
 * @LastEditTime: 2023-12-04 19:55:58
 */
#include "MPCSolver.h"

MPCSolver::MPCSolver(int &nx, int &nu, int &N,
                     Eigen::VectorXd &x0, Eigen::VectorXd &u0, Eigen::MatrixXd &xr,
                     Eigen::VectorXd &xmin, Eigen::VectorXd &xmax,
                     Eigen::VectorXd &umin, Eigen::VectorXd &umax,
                     Eigen::MatrixXd &Q, Eigen::MatrixXd &QN, Eigen::MatrixXd &R,
                     Eigen::MatrixXd &A, Eigen::VectorXd &B) : nx(nx), nu(nu), N(N), x0(x0), u0(u0), xr(xr), xmin(xmin), xmax(xmax), umin(umin), umax(umax), Q(Q), QN(QN), R(R), A(A), B(B){};

// 设置代价函数二次项系数矩阵
void MPCSolver::SetHessianMatrix(Eigen::SparseMatrix<double> &hessian)
{
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(N, N);
    Eigen::MatrixXd P1 = kroneckerProduct(I, Q);
    Eigen::MatrixXd P2 = kroneckerProduct(I, R);
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero((P1.rows() + QN.rows() + P2.rows()), (P1.cols() + QN.cols() + P2.cols()));
    P.block(0, 0, P1.rows(), P1.cols()) = P1;
    P.block(P1.rows(), P1.cols(), QN.rows(), QN.cols()) = QN;
    P.block(P1.rows() + QN.rows(), P1.cols() + QN.cols(), P2.rows(), P2.cols()) = P2;
    hessian = P.sparseView(); // 转换为稀疏矩阵
    // std::cout << P << std::endl;
};

// 设置代价函数一次项系数
void MPCSolver::SetGradientVector(Eigen::VectorXd &gradient)
{
    Eigen::VectorXd q = Eigen::VectorXd::Zero((N + 1) * nx + N * nu);
    int start_index = 0;
    for (size_t i = 0; i < N - 1; i++)
    {
        q.segment(start_index, nx) = -Q * xr.row(i).transpose();
        start_index += nx;
    }
    q.segment((N - 1) * nx, nx) = -Q * xr.row(N - 1).transpose();

    gradient = q;
};

// 设置约束方程系数矩阵和约束上下界
void MPCSolver::SetConstraintMatrix(Eigen::VectorXd &lowerBound, Eigen::VectorXd &upperBound,
                                    Eigen::SparseMatrix<double> &linearMatrix)
{
    // 等式约束
    Eigen::MatrixXd I1 = Eigen::MatrixXd::Identity(N + 1, N + 1);
    Eigen::MatrixXd I2 = Eigen::MatrixXd::Identity(nx, nx);
    Eigen::MatrixXd I3 = Eigen::MatrixXd::Zero(N + 1, N + 1);
    for (size_t i = 1; i < N + 1; i++) // 注意，要检索第一条主对角线的元素，i要从1开始，因为起始位置为(1,0)
    {
        I3(i, i - 1) = 1;
    }
    Eigen::MatrixXd A1 = kroneckerProduct(I1, -I2);
    Eigen::MatrixXd A2 = kroneckerProduct(I3, A);
    Eigen::MatrixXd Ax = A1 + A2;

    Eigen::MatrixXd I4 = I1.block(0, 1, N + 1, N);
    Eigen::MatrixXd Bu = kroneckerProduct(I4, B);

    Eigen::MatrixXd Aeq((N + 1) * nx, (N + 1) * nx + N * nu);
    Aeq << Ax, Bu;

    Eigen::VectorXd leq = Eigen::VectorXd::Zero((N + 1) * nx);
    for (size_t j = 0; j < nx; j++)
    {
        leq(j) = -x0(j);
    }
    Eigen::VectorXd ueq = leq;

    // 不等式约束
    Eigen::MatrixXd Aineq = Eigen::MatrixXd::Identity(nx * (N + 1) + nu * N, nx * (N + 1) + nu * N);

    Eigen::VectorXd I5 = Eigen::VectorXd::Ones(N + 1);
    Eigen::VectorXd I6 = Eigen::VectorXd::Ones(N);
    Eigen::VectorXd lineq1 = kroneckerProduct(I5, xmin);
    Eigen::VectorXd lineq2 = kroneckerProduct(I6, umin);
    Eigen::VectorXd lineq(nx * (N + 1) + nu * N);
    lineq << lineq1, lineq2;
    Eigen::VectorXd uineq1 = kroneckerProduct(I5, xmax);
    Eigen::VectorXd uineq2 = kroneckerProduct(I6, umax);
    Eigen::VectorXd uineq(nx * (N + 1) + nu * N);
    uineq << uineq1, uineq2;

    // 约束汇总
    Eigen::MatrixXd G(2 * nx * (N + 1) + nu * N, nx * (N + 1) + nu * N);
    G << Aeq,
        Aineq;
    Eigen::VectorXd l(2 * nx * (N + 1) + nu * N);
    l << leq, lineq;
    Eigen::VectorXd u(2 * nx * (N + 1) + nu * N);
    u << ueq, uineq;

    linearMatrix = G.sparseView();
    lowerBound = l;
    upperBound = u;
};