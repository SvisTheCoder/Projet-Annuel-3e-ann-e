#include <iostream>
#include <Eigen/Dense>

#include "perceptron.hpp"
#include "mlp.hpp"
#include "rbf.hpp"
#include "svm.hpp"

int main() {
    Eigen::MatrixXd X(8, 2);

    X << 1.0, 1.0,
         1.5, 2.0,
         2.0, 1.0,
         2.0, 2.5,
         5.0, 5.0,
         6.0, 5.0,
         5.5, 6.0,
         6.0, 6.5;

    Eigen::VectorXi yPerceptron(8);
    yPerceptron << -1, -1, -1, -1, 1, 1, 1, 1;

    Eigen::VectorXi yBinary(8);
    yBinary << 0, 0, 0, 0, 1, 1, 1, 1;

    std::cout << "=== PERCEPTRON ===" << std::endl;
    Perceptron perceptron(0.1, 20);
    perceptron.fit(X, yPerceptron);

    std::cout << "Accuracy : " << perceptron.score(X, yPerceptron) << std::endl;
    std::cout << "Weights : " << perceptron.getWeights().transpose() << std::endl;
    std::cout << "Bias : " << perceptron.getBias() << std::endl;

    std::cout << "\n MLP" << std::endl;
    MLP mlp(2, 4, 0.1, 2000);
    mlp.fit(X, yBinary);

    std::cout << "Accuracy : " << mlp.score(X, yBinary) << std::endl;

    std::cout << "\n RBF" << std::endl;
    RBF rbf(4, 1.0, 0.1, 1000);
    rbf.fit(X, yBinary);

    std::cout << "Accuracy : " << rbf.score(X, yBinary) << std::endl;

    std::cout << "\n SVM" << std::endl;
    SVM svm(0.01, 1000, 0.001);
    svm.fit(X, yPerceptron);

    std::cout << "Accuracy : " << svm.score(X, yPerceptron) << std::endl;
    std::cout << "Weights : " << svm.getWeights().transpose() << std::endl;
    std::cout << "Bias : " << svm.getBias() << std::endl;

    return 0;
}