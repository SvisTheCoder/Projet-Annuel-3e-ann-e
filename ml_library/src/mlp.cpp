#include "mlp.hpp"
#include <cmath>

MLP::MLP(int inputSize, int hiddenSize, double learningRate, int epochs) {
    this->inputSize = inputSize;
    this->hiddenSize = hiddenSize;
    this->learningRate = learningRate;
    this->epochs = epochs;

    W1 = Eigen::MatrixXd::Random(inputSize, hiddenSize) * 0.5;
    b1 = Eigen::VectorXd::Zero(hiddenSize);

    W2 = Eigen::VectorXd::Random(hiddenSize) * 0.5;
    b2 = 0.0;
}

double MLP::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double MLP::sigmoidDerivative(double value) const {
    return value * (1.0 - value);
}

void MLP::fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    lossHistory.clear();

    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;

        for (int i = 0; i < X.rows(); i++) {
            Eigen::VectorXd x = X.row(i);
            double target = y(i);

            Eigen::VectorXd z1 = W1.transpose() * x + b1;
            Eigen::VectorXd a1 = z1.unaryExpr([this](double v) { return sigmoid(v); });

            double z2 = W2.dot(a1) + b2;
            double prediction = sigmoid(z2);

            double errorOutput = prediction - target;

            totalLoss += errorOutput * errorOutput;

            Eigen::VectorXd errorHidden(hiddenSize);

            for (int h = 0; h < hiddenSize; h++) {
                errorHidden(h) = errorOutput * W2(h) * sigmoidDerivative(a1(h));
            }

            W2 = W2 - learningRate * errorOutput * a1;
            b2 = b2 - learningRate * errorOutput;

            W1 = W1 - learningRate * x * errorHidden.transpose();
            b1 = b1 - learningRate * errorHidden;
        }

        lossHistory.push_back(totalLoss / X.rows());
    }
}

double MLP::predictProba(const Eigen::VectorXd& x) const {
    Eigen::VectorXd z1 = W1.transpose() * x + b1;
    Eigen::VectorXd a1 = z1.unaryExpr([this](double v) { return sigmoid(v); });

    double z2 = W2.dot(a1) + b2;
    return sigmoid(z2);
}

int MLP::predict(const Eigen::VectorXd& x) const {
    if (predictProba(x) >= 0.5) {
        return 1;
    }

    return 0;
}

double MLP::score(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) const {
    int correct = 0;

    for (int i = 0; i < X.rows(); i++) {
        if (predict(X.row(i)) == y(i)) {
            correct++;
        }
    }

    return (double)correct / X.rows();
}

std::vector<double> MLP::getLossHistory() const {
    return lossHistory;
}