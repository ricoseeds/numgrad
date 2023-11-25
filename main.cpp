#include <iostream>
#include <Eigen/Dense>

int main()
{
    Eigen::Vector3d x(1.0, 2.0, 3.0);
    Eigen::Vector3d y;
    y = x;

    y(0) = 11;
    x(0) = 2;
    Eigen::MatrixXd J = Eigen::MatrixXd::Identity(3, 3);
    J(0, 1) = 2;
    J(0, 2) = 3;
    J(1, 0) = 82;
    J(1, 2) = 5;
    J(2, 1) = 2;
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(3, 3);
    H(1, 0) = 3;

    // Eigen::MatrixXd result(2 * J.rows(), 2 * J.cols());
    // result.block(0, 0, J.rows(), J.cols()) = J;
    // result.block(0, J.cols(), J.rows(), J.cols()) = -J;
    // result.block(J.rows(), 0, J.rows(), J.cols()) = -J;
    // result.block(J.rows(), J.cols(), J.rows(), J.cols()) = J;
    // std::cout << "Result Matrix:\n"
    //   << result << std::endl;

    std::cout << J * H << std::endl;
    // std::cout << x << std::endl;

    // // std::cout << x << "\n";
    // // std::cout << (x - y).norm() << "\n";
    // Eigen::VectorXd f;
    // f = Eigen::VectorXd(10);
    // f.setConstant(1.0);
    // // std::cout << f.rows() << "\n";
    // f.segment<3>(0) = Eigen::Vector3d(1.6, 3.4, 54.5);
    // std::cout << f.segment<3>(2) << "\n";
    // Eigen::VectorXf fc = f.cast<float>();
    // std::cout << "converted\n"
    //           << fc;
    // std::cout << std::endl;
    // Eigen::MatrixXd m = Eigen::MatrixXd(3, 3);
    // Eigen::Vector3d r = x / x.norm();
    // Eigen::MatrixXd t = x * x.transpose();
    // Eigen::Matrix3d mm = Eigen::Matrix3d::Identity();
    // mm = mm / 3;
    // std::cout << "mat = " << mm;
    // std::cout << "Norm" << (x / x.norm()).norm();

    // std::cout << r << "\n";

    return 0;
}