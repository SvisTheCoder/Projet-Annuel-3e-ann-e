#include "rbf.hpp"
#include <cmath>

RBF::RBF(int numCenters, double sigma, double learningRate, int epochs) {
    this->numCenters = numCenters;
    this->sigma = sigma;
    this->learningRate = learningRate;
    this->epochs = epochs;

    bias = 0.0;
}

double RBF::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double RBF::activation(const Eigen::VectorXd& x, const Eigen::VectorXd& center) const {
    double distanceSquared = (x - center).squaredNorm();
    return std::exp(-distanceSquared / (2.0 * sigma * sigma));
}

void RBF::fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    int inputSize = X.cols();

    centers = Eigen::MatrixXd(numCenters, inputSize);
    weights = Eigen::VectorXd::Zero(numCenters);

    for (int c = 0; c < numCenters; c++) {
        centers.row(c) = X.row(c % X.rows());
    }

    errorsPerEpoch.clear();

    for (int epoch = 0; epoch < epochs; epoch++) {
        int errors = 0;

        for (int i = 0; i < X.rows(); i++) {
            Eigen::VectorXd x = X.row(i);
            Eigen::VectorXd activations(numCenters);

            for (int c = 0; c < numCenters; c++) {
                activations(c) = activation(x, centers.row(c));
            }

            double output = weights.dot(activations) + bias;
            double probability = sigmoid(output);

            int prediction = probability >= 0.5 ? 1 : 0;

            if (prediction != y(i)) {
                errors++;
            }

            double error = probability - y(i);

            weights = weights - learningRate * error * activations;
            bias = bias - learningRate * error;
        }

        errorsPerEpoch.push_back(errors);
    }
}

double RBF::predictProba(const Eigen::VectorXd& x) const {
    Eigen::VectorXd activations(numCenters);

    for (int c = 0; c < numCenters; c++) {
        activations(c) = activation(x, centers.row(c));
    }

    double output = weights.dot(activations) + bias;
    return sigmoid(output);
}

int RBF::predict(const Eigen::VectorXd& x) const {
    if (predictProba(x) >= 0.5) {
        return 1;
    }

    return 0;
}

double RBF::score(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) const {
    int correct = 0;

    for (int i = 0; i < X.rows(); i++) {
        if (predict(X.row(i)) == y(i)) {
            correct++;
        }
    }

    return (double)correct / X.rows();
}

std::vector<int> RBF::getErrors() const {
    return errorsPerEpoch;
}