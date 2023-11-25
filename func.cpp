#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

// Define your function
double myFunction(const vector<double> &x1, const vector<double> &x2)
{
    // Example function: f(x1, x2) = x1^2 + x2^2
    double result = 0.0;
    for (int i = 0; i < 3; ++i)
    {
        result += x1[i] * x1[i] + x2[i] * x2[i];
    }
    return result;
}

// Compute the numerical gradient of a function
void numericalGradient(const vector<double> &x1, const vector<double> &x2, vector<double> &gradient)
{
    const double epsilon = 1e-6;

    for (int i = 0; i < 3; ++i)
    {
        vector<double> x1_plus_delta = x1;
        vector<double> x1_minus_delta = x1;

        x1_plus_delta[i] += epsilon;
        x1_minus_delta[i] -= epsilon;

        double gradient_i = (myFunction(x1_plus_delta, x2) - myFunction(x1_minus_delta, x2)) / (2 * epsilon);

        gradient[i] = gradient_i;
    }
}

// Compute the numerical Hessian matrix of a function
void numericalHessian(const vector<double> &x1, const vector<double> &x2, vector<vector<double>> &hessian)
{
    const double epsilon = 1e-6;

    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            vector<double> x1_plus_delta = x1;
            vector<double> x1_minus_delta = x1;
            vector<double> x2_plus_delta = x2;
            vector<double> x2_minus_delta = x2;

            x1_plus_delta[i] += epsilon;
            x1_minus_delta[i] -= epsilon;

            x2_plus_delta[j] += epsilon;
            x2_minus_delta[j] -= epsilon;

            double hessian_ij =
                (myFunction(x1_plus_delta, x2_plus_delta) - myFunction(x1_plus_delta, x2_minus_delta) -
                 myFunction(x1_minus_delta, x2_plus_delta) + myFunction(x1_minus_delta, x2_minus_delta)) /
                (4 * epsilon * epsilon);

            hessian[i][j] = hessian_ij;
        }
    }
}

int main()
{
    // Example usage
    vector<double> x1 = {1.0, 2.0, 3.0};
    vector<double> x2 = {4.0, 5.0, 6.0};

    // Numerical gradient
    vector<double> gradient(3, 0.0);
    numericalGradient(x1, x2, gradient);

    // Numerical Hessian
    vector<vector<double>> hessian(3, vector<double>(3, 0.0));
    numericalHessian(x1, x2, hessian);

    // Output results
    cout << "Numerical Gradient:" << endl;
    for (double g : gradient)
    {
        cout << g << " ";
    }
    cout << endl;

    cout << "Numerical Hessian:" << endl;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            cout << hessian[i][j] << " ";
        }
        cout << endl;
    }

    return 0;
}
