#include "mlp.hpp"

#include <cmath>

MLP::MLP(
    int inputSize,
    int hiddenSize,
    double learningRate,
    int epochs
) {
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
    if (X.rows() == 0 || X.rows() != y.size()) {
        return;
    }

    lossHistory.clear();

    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0.0;

        for (int i = 0; i < X.rows(); i++) {
            Eigen::VectorXd x = X.row(i).transpose();
            double target = static_cast<double>(y(i));

            Eigen::VectorXd z1 = W1.transpose() * x + b1;

            Eigen::VectorXd a1 = z1.unaryExpr(
                [this](double value) {
                    return sigmoid(value);
                }
            );

            double z2 = W2.dot(a1) + b2;
            double prediction = sigmoid(z2);

            double errorOutput = prediction - target;
            totalLoss += errorOutput * errorOutput;

            Eigen::VectorXd errorHidden(hiddenSize);

            for (int h = 0; h < hiddenSize; h++) {
                errorHidden(h) =
                    errorOutput
                    * W2(h)
                    * sigmoidDerivative(a1(h));
            }

            W2 -= learningRate * errorOutput * a1;
            b2 -= learningRate * errorOutput;

            W1 -= learningRate * x * errorHidden.transpose();
            b1 -= learningRate * errorHidden;
        }

        lossHistory.push_back(
            totalLoss / static_cast<double>(X.rows())
        );
    }
}

double MLP::predictProba(const Eigen::VectorXd& x) const {
    Eigen::VectorXd z1 = W1.transpose() * x + b1;

    Eigen::VectorXd a1 = z1.unaryExpr(
        [this](double value) {
            return sigmoid(value);
        }
    );

    double z2 = W2.dot(a1) + b2;

    return sigmoid(z2);
}

int MLP::predict(const Eigen::VectorXd& x) const {
    if (predictProba(x) >= 0.5) {
        return 1;
    }

    return 0;
}

double MLP::score(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXi& y
) const {
    if (X.rows() == 0 || X.rows() != y.size()) {
        return 0.0;
    }

    int correct = 0;

    for (int i = 0; i < X.rows(); i++) {
        Eigen::VectorXd x = X.row(i).transpose();

        if (predict(x) == y(i)) {
            correct++;
        }
    }

    return static_cast<double>(correct)
        / static_cast<double>(X.rows());
}

std::vector<double> MLP::getLossHistory() const {
    return lossHistory;
}

const Eigen::MatrixXd& MLP::getInputWeights() const { return W1; }
const Eigen::VectorXd& MLP::getHiddenBias() const { return b1; }
const Eigen::VectorXd& MLP::getOutputWeights() const { return W2; }
double MLP::getOutputBias() const { return b2; }

void MLP::setParameters(
    const Eigen::MatrixXd& newW1,
    const Eigen::VectorXd& newB1,
    const Eigen::VectorXd& newW2,
    double newB2
) {
    W1 = newW1;
    b1 = newB1;
    W2 = newW2;
    b2 = newB2;
}
