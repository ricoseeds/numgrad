#include <iostream>
#include <cmath>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <iomanip> // std::setprecision
double k = 0.1;
double l0 = 1.0;

enum AccuracyOrder
{
    SECOND, ///< @brief Second order accuracy.
    FOURTH, ///< @brief Fourth order accuracy.
    SIXTH,  ///< @brief Sixth order accuracy.
    EIGHTH  ///< @brief Eighth order accuracy.
};
std::vector<double> get_external_coeffs(const AccuracyOrder accuracy)
{
    switch (accuracy)
    {
    case SECOND:
        return {{1, -1}};
    case FOURTH:
        return {{1, -8, 8, -1}};
    case SIXTH:
        return {{-1, 9, -45, 45, -9, 1}};
    case EIGHTH:
        return {{3, -32, 168, -672, 672, -168, 32, -3}};
    default:
        throw std::invalid_argument("invalid accuracy order");
    }
}

std::vector<double> get_interior_coeffs(const AccuracyOrder accuracy)
{
    switch (accuracy)
    {
    case SECOND:
        return {{1, -1}};
    case FOURTH:
        return {{-2, -1, 1, 2}};
    case SIXTH:
        return {{-3, -2, -1, 1, 2, 3}};
    case EIGHTH:
        return {{-4, -3, -2, -1, 1, 2, 3, 4}};
    default:
        throw std::invalid_argument("invalid accuracy order");
    }
}
double get_denominator(const AccuracyOrder accuracy)
{
    switch (accuracy)
    {
    case SECOND:
        return 2;
    case FOURTH:
        return 12;
    case SIXTH:
        return 60;
    case EIGHTH:
        return 840;
    default:
        throw std::invalid_argument("invalid accuracy order");
    }
}

double func(Eigen::Vector3d x1, Eigen::Vector3d x2)
{

    Eigen::Vector3d x12 = x1 - x2;
    double x12norm = x12.norm();
    double r_ = (x12norm / l0) - 1.0;
    double e = (k / 2.0) * (r_ * r_);
    return e;
    // Eigen::Vector3d x12 = x1 - x2;
    // double ret = x12.norm();
    // return ret;
}
void finite_gradient(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const std::function<double(const Eigen::VectorXd &)> &f,
    Eigen::VectorXd &grad,
    const AccuracyOrder accuracy,
    const double eps = 1.0e-8)
{
    const std::vector<double> external_coeffs = get_external_coeffs(accuracy);
    const std::vector<double> internal_coeffs = get_interior_coeffs(accuracy);

    assert(external_coeffs.size() == internal_coeffs.size());
    const size_t inner_steps = internal_coeffs.size();

    const double denom = get_denominator(accuracy) * eps;

    grad.setZero(x.rows());

    Eigen::VectorXd x_mutable = x;
    for (size_t i = 0; i < x.rows(); i++)
    {
        for (size_t ci = 0; ci < inner_steps; ci++)
        {
            x_mutable[i] += internal_coeffs[ci] * eps;
            grad[i] += external_coeffs[ci] * f(x_mutable);
            x_mutable[i] = x[i];
        }
        grad[i] /= denom;
    }
}
void finite_hessian(
    const Eigen::Ref<const Eigen::VectorXd> &x,
    const std::function<double(const Eigen::VectorXd &)> &f,
    Eigen::MatrixXd &hess,
    const AccuracyOrder accuracy,
    const double eps = 1.0e-8)
{
    const std::vector<double> external_coeffs = get_external_coeffs(accuracy);
    const std::vector<double> internal_coeffs = get_interior_coeffs(accuracy);

    assert(external_coeffs.size() == internal_coeffs.size());
    const size_t inner_steps = internal_coeffs.size();

    double denom = get_denominator(accuracy) * eps;
    denom *= denom;

    hess.setZero(x.rows(), x.rows());

    Eigen::VectorXd x_mutable = x;
    for (size_t i = 0; i < x.rows(); i++)
    {
        for (size_t j = i; j < x.rows(); j++)
        {
            for (size_t ci = 0; ci < inner_steps; ci++)
            {
                for (size_t cj = 0; cj < inner_steps; cj++)
                {
                    x_mutable[i] += internal_coeffs[ci] * eps;
                    x_mutable[j] += internal_coeffs[cj] * eps;
                    hess(i, j) += external_coeffs[ci] * external_coeffs[cj] * f(x_mutable);
                    x_mutable[j] = x[j];
                    x_mutable[i] = x[i];
                }
            }
            hess(i, j) /= denom;
            hess(j, i) = hess(i, j); // The hessian is symmetric
        }
    }
}

Eigen::VectorXd g(Eigen::Vector3d xi, Eigen::Vector3d xj)
{

    Eigen::Vector3d xij = xi - xj;
    double xijnorm = xij.norm();
    Eigen::Vector3d xijhat = xij / xijnorm;
    double multiplier = (-k / l0) * ((xijnorm / l0) - 1.0);
    Eigen::Vector3d frc = multiplier * xijhat;
    return frc;
}
// Compute the numerical gradient of a function
Eigen::VectorXd grad(Eigen::Vector3d &x1, Eigen::Vector3d &x2)
{
    Eigen::VectorXd gradient = Eigen::VectorXd(6);
    const double epsilon = 1e-6;
    Eigen::VectorXd x12 = Eigen::VectorXd(6);
    x12.segment<3>(0) = x1;
    x12.segment<3>(3) = x2;
    Eigen::VectorXd x12save = x12;
    for (int i = 0; i < 6; ++i)
    {
        x12save(i) += epsilon;
        gradient(i) = (func(x12save.segment<3>(0), x12save.segment<3>(3)) - func(x1, x2)) / epsilon;
        x12save(i) -= epsilon;
    }
    return gradient;
}

// Compute the numerical gradient of a function
Eigen::MatrixXd hess(Eigen::Vector3d &x1, Eigen::Vector3d &x2)
{
    Eigen::MatrixXd hessian = Eigen::MatrixXd(6, 6);
    const double epsilon = 1e-6;
    Eigen::VectorXd x12 = Eigen::VectorXd(6);
    x12.segment<3>(0) = x1;
    x12.segment<3>(3) = x2;
    Eigen::VectorXd x12save = x12;
    for (int i = 0; i < 6; ++i)
    {
        x12save(i) += epsilon;
        for (int j = 0; j < 6; j++)
        {
            x12save(j) += epsilon;
            hessian(i, j) = (func(x12save.segment<3>(0), x12save.segment<3>(3)) - func(x1, x2)) / epsilon;
            x12save(j) -= epsilon;
        }
        x12save(i) -= epsilon;
    }
    return hessian;
}

int main()
{
    // std::cout << std::setprecision(10);
    Eigen::Vector3d x1 = Eigen::Vector3d(1, 2, 3);
    Eigen::Vector3d x2 = Eigen::Vector3d(2, 4, 6);
    Eigen::VectorXd g = grad(x1, x2);
    std::cout << "\n my grad \n"
              << grad(x1, x2);
    // // std::cout << g.segment<3>(3).norm();
    // // std::cout << "func " << func(x1, x2);
    // std::cout << "\n g \n " << g(x1, x2);
    // // std::cout << "g" << g(x1, x2);
    std::cout << "\n hess\n " << hess(x1, x2);
    const auto f = [&](const Eigen::VectorXd &x) -> double
    {
        return func(x.segment<3>(0), x.segment<3>(3));
    };
    Eigen::VectorXd x = Eigen::VectorXd(6);
    x.segment<3>(0) = x1.segment<3>(0);
    x.segment<3>(3) = x2.segment<3>(0);
    Eigen::VectorXd fgrad;
    Eigen::MatrixXd fhess;
    finite_gradient(x, f, fgrad, SECOND);
    std::cout << "\n grad \n"
              << fgrad;
    finite_hessian(x, f, fhess, SECOND);
    std::cout << "\n hess \n"
              << fhess;

    return 0;
}