/*
 * @Description:
 * @Version: 2.0
 * @Author: ZHAO B.T.
 * @Date: 2023-11-23 12:01:15
 * @LastEditors: ZHAO B.T.
 * @LastEditTime: 2023-12-04 10:35:57
 */
#ifndef MPCSolver_H
#define MPCSolver_H

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include </home/zbt/lib/eigen-3.4.0/unsupported/Eigen/KroneckerProduct>
#include "OsqpEigen/OsqpEigen.h"
#include <osqp/osqp.h>
#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class MPCSolver
{
private:
    int nx, nu, N;        // 状态变量维度，控制变量维度，预测时域
    Eigen::VectorXd x0;   // 状态变量初始值
    Eigen::VectorXd u0;   // 控制变量初始值
    Eigen::MatrixXd xr;   // 状态变量参考值
    Eigen::VectorXd xmin; // 状态变量约束
    Eigen::VectorXd xmax;
    Eigen::VectorXd umin; // 控制变量约束
    Eigen::VectorXd umax;
    Eigen::MatrixXd Q;  // 状态变量误差矩阵
    Eigen::MatrixXd QN; // 状态变量末态误差矩阵
    Eigen::MatrixXd R;  // 控制变量误差矩阵
    Eigen::MatrixXd A;  // 状态矩阵
    Eigen::MatrixXd B;  // 控制矩阵
public:
    MPCSolver(int &nx, int &nu, int &N,
              Eigen::VectorXd &x0, Eigen::VectorXd &u0, Eigen::MatrixXd &xr,
              Eigen::VectorXd &xmin, Eigen::VectorXd &xmax,
              Eigen::VectorXd &umin, Eigen::VectorXd &umax,
              Eigen::MatrixXd &Q, Eigen::MatrixXd &QN, Eigen::MatrixXd &R,
              Eigen::MatrixXd &A, Eigen::VectorXd &B);
    ~MPCSolver(){}; // 构造函数带参，析构函数也必须带参，即有{}

    void SetHessianMatrix(Eigen::SparseMatrix<double> &hessian);
    void SetGradientVector(Eigen::VectorXd &gradient);
    void SetConstraintMatrix(Eigen::VectorXd &lowerBound, Eigen::VectorXd &upperBound,
                             Eigen::SparseMatrix<double> &linearMatrix);
};

#endif