/*
 * @Description:
 * @Version: 2.0
 * @Author: ZHAO B.T.
 * @Date: 2023-11-23 09:33:52
 * @LastEditors: ZHAO B.T.
 * @LastEditTime: 2023-12-04 20:12:06
 */

// 自车信息
#define MAX_SPEED 40 / 3.6
#define MIN_SPEED 0 / 3.6
#define MAX_D 100 // 最大车距
#define MIN_D 5   // 最小车距
#define MAX_REL_SPEED 40 / 3.6
#define MIN_REL_SPEED -40 / 3.6
#define MAX_ACC 2.5
#define MIN_ACC -5.0
#define MAX_JERK 2.5
#define MIN_JERK -2.5
#define TARGET_SPEED 20 / 3.6
#define TARGET_D 25 // 目标车距
#define DT 0.2
#define S0_r 0           // 后车初始位置
#define Speed0_r 0 / 3.6 // 后车初始车速

// 前车信息
#define S0_f 20          // 前车初始位置
#define Speed_f 20 / 3.6 // 前车车速

#include "MPCSolver.h"

int main()
{
    // // 模型参数矩阵
    // Eigen::MatrixXd Q(5, 5); // 状态变量误差系数矩阵,v,d,vr,a,jerk
    // Q << 1, 0, 0, 0, 0,
    //     0, 1, 0, 0, 0,
    //     0, 0, 1, 0, 0,
    //     0, 0, 0, 0, 0,
    //     0, 0, 0, 0, 1;
    // Eigen::MatrixXd R(1, 1); // 控制变量误差矩阵
    // R << 1;
    // Eigen::MatrixXd QN = Q;  // 末状态状态变量系数矩阵
    // Eigen::MatrixXd A(5, 5); // 状态矩阵
    // A << 1, 0, 0, DT, 0,
    //     0, 1, DT, -0.5 * DT * DT, 0,
    //     0, 0, 1, -DT, 0,
    //     0, 0, 0, -2 * DT, 0,
    //     0, 0, 0, -2, 0;

    // Eigen::VectorXd B(5); // 控制矩阵
    // B << 0, 0, 0, 0.4, 2;

    // // 初始值
    // Eigen::VectorXd x0(5); // v,d,vr,a,jerk
    // x0 << 0, 20, 20 / 3.6, 0, 0;
    // int nx = x0.size();
    // Eigen::VectorXd xr(5);
    // xr << TARGET_SPEED, TARGET_D, 0, 0, 0;
    // Eigen::VectorXd u0(1);
    // u0 << 0;
    // int nu = u0.size();

    // // 约束
    // Eigen::VectorXd umin(1);
    // umin << MIN_ACC;
    // Eigen::VectorXd umax(1);
    // umax << MAX_ACC;
    // Eigen::VectorXd xmin(5);
    // xmin << MIN_SPEED, MIN_D, MIN_REL_SPEED, MIN_ACC, MIN_JERK;
    // Eigen::VectorXd xmax(5);
    // xmax << MAX_SPEED, MAX_D, MAX_REL_SPEED, MAX_ACC, MAX_JERK;

    // 模型参数矩阵
    Eigen::MatrixXd Q(3, 3); // 状态变量误差系数矩阵,v,d,vr
    Q << 1, 0, 0,
        0, 0, 0,
        0, 0, 1;

    Eigen::MatrixXd R(1, 1); // 控制变量误差矩阵
    R << 1;

    Eigen::MatrixXd QN = Q; // 末状态状态变量系数矩阵

    Eigen::MatrixXd A(3, 3); // 状态矩阵
    A << 1, 0, 0,
        0, 1, DT,
        0, 0, 1;

    Eigen::VectorXd B(3); // 控制矩阵
    B << DT, -0.5 * DT * DT, -DT;

    // 初始值
    Eigen::VectorXd x0(3); // v,d,vr
    x0 << Speed0_r, S0_f - S0_r, Speed_f - Speed0_r;
    int nx = x0.size();

    Eigen::VectorXd u0(1);
    u0 << 0;
    int nu = u0.size();

    // 约束
    Eigen::VectorXd umin(1);
    umin << MIN_ACC;
    Eigen::VectorXd umax(1);
    umax << MAX_ACC;
    Eigen::VectorXd xmin(3);
    xmin << MIN_SPEED, MIN_D, MIN_REL_SPEED;
    Eigen::VectorXd xmax(3);
    xmax << MAX_SPEED, MAX_D, MAX_REL_SPEED;

    // 预测时域
    int N = 10;

    // 状态变量参考值
    Eigen::MatrixXd xr(N, nx);
    Eigen::VectorXd Vec_xr(nx);
    Vec_xr << TARGET_SPEED, TARGET_D, 0;
    for (size_t i = 0; i < N; i++)
    {
        xr.row(i) = Vec_xr;
    }

    // 仿真参数初始化
    float S_f = S0_f;
    float S_r = S0_r;
    float Speed_r = Speed0_r;
    std::vector<float> Vec_SpeedFrontCar{Speed_f};
    std::vector<float> Vec_SpeedRearCar{Speed0_r};
    std::vector<float> Vec_t{0};
    std::vector<float> Vec_D{S0_f - S0_r};
    std::vector<float> Vec_Acc{0};

    // MPC求解器求解QP问题，每个sim循环周期为0.2s
    for (int sim = 0; sim < 400; sim++)
    {

        MPCSolver FollowVehicle(nx, nu, N, x0, u0, xr, xmin, xmax, umin, umax, Q, QN, R, A, B);

        Eigen::SparseMatrix<double> hessian;
        Eigen::VectorXd gradient;
        Eigen::SparseMatrix<double> linearMatrix;
        Eigen::VectorXd lowerBound;
        Eigen::VectorXd upperBound;

        FollowVehicle.SetHessianMatrix(hessian);
        FollowVehicle.SetGradientVector(gradient);
        FollowVehicle.SetConstraintMatrix(lowerBound, upperBound, linearMatrix);

        OsqpEigen::Solver solver;
        solver.settings()->setVerbosity(false); // 求解信息可视化
        solver.settings()->setWarmStart(true);
        solver.data()->setNumberOfVariables(hessian.cols());
        solver.data()->setNumberOfConstraints(linearMatrix.rows());

        if (!solver.data()->setHessianMatrix(hessian))
            return false;
        if (!solver.data()->setLinearConstraintsMatrix(linearMatrix))
            return false;
        if (!solver.data()->setGradient(gradient))
            return false; // 注意，一次项系数set必须为一维数组，不能为矩阵
        if (!solver.data()->setLowerBound(lowerBound))
            return false;
        if (!solver.data()->setUpperBound(upperBound))
            return false;
        if (!solver.initSolver())
            return false;
        if (static_cast<int>(solver.solveProblem()) != 0)
            return false;

        Eigen::VectorXd output = solver.getSolution();
        // std::cout << output.size() << std::endl;

        // 状态更新
        // 前车状态更新
        S_f = S_f + Speed_f * DT;
        // 后车状态更新
        float AccNew = output((N + 1) * nx);
        Speed_r = Speed_r + AccNew * DT;
        S_r = S_r + Speed_r * DT + 0.5 * AccNew * DT * DT;

        x0(0) = Speed_r;
        x0(1) = S_f - S_r;
        x0(2) = Speed_f - Speed_r;

        Vec_SpeedFrontCar.push_back(Speed_f);
        Vec_SpeedRearCar.push_back(Speed_r);
        Vec_D.push_back(S_f - S_r);
        Vec_t.push_back((sim + 1) * DT);
        Vec_Acc.push_back(AccNew);

        std::cout << " 时间：" << (sim + 1) * DT
                  << " 前车车速：" << Speed_f
                  << " 自车车速：" << Speed_r
                  << " 加速度：" << AccNew
                  << " 前车位置：" << S_f
                  << " 后车位置：" << S_r
                  << " 车距：" << S_f - S_r
                  << std::endl;
    }
    // 绘图
    plt::figure_size(1500, 500);
    plt::subplot(1, 3, 1);
    plt::plot(Vec_t, Vec_SpeedFrontCar, "r-",
              Vec_t, Vec_SpeedRearCar, "b-");
    plt::xlim(0, int(Vec_t.back()));
    plt::ylim(0, 15);
    plt::xlabel("t (s)");
    plt::ylabel("Vel (km/h)");
    plt::title("Vel-t");
    plt::grid(1);

    plt::subplot(1, 3, 2);
    plt::plot(Vec_t, Vec_Acc, "b-");
    plt::xlim(0, int(Vec_t.back()));
    plt::ylim(-10, 10);
    plt::xlabel("t (s)");
    plt::ylabel("Acc (m/s2)");
    plt::title("Acc-t");
    plt::grid(1);

    plt::subplot(1, 3, 3);
    plt::plot(Vec_t, Vec_D, "r-");
    plt::xlim(0, int(Vec_t.back()));
    plt::ylim(0, 30);
    plt::xlabel("t (s)");
    plt::ylabel("D (m)");
    plt::title("D-t");
    plt::grid(1);
    plt::show();
    return 0;
}