#include "rbf.hpp"

#include <cmath>

RBF::RBF(
    int numCenters,
    double sigma,
    double learningRate,
    int epochs
) {
    this->numCenters = numCenters;
    this->sigma = sigma;
    this->learningRate = learningRate;
    this->epochs = epochs;

    bias = 0.0;
}

double RBF::sigmoid(double x) const {
    return 1.0 / (1.0 + std::exp(-x));
}

double RBF::activation(
    const Eigen::VectorXd& x,
    const Eigen::VectorXd& center
) const {
    double distanceSquared = (x - center).squaredNorm();

    return std::exp(
        -distanceSquared / (2.0 * sigma * sigma)
    );
}

void RBF::fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    if (X.rows() == 0 || X.rows() != y.size()) {
        return;
    }

    int inputSize = X.cols();

    centers = Eigen::MatrixXd(numCenters, inputSize);
    weights = Eigen::VectorXd::Zero(numCenters);
    bias = 0.0;

    for (int c = 0; c < numCenters; c++) {
        centers.row(c) = X.row(c % X.rows());
    }

    errorsPerEpoch.clear();

    for (int epoch = 0; epoch < epochs; epoch++) {
        int errors = 0;

        for (int i = 0; i < X.rows(); i++) {
            Eigen::VectorXd x = X.row(i).transpose();
            Eigen::VectorXd activations(numCenters);

            for (int c = 0; c < numCenters; c++) {
                activations(c) = activation(
                    x,
                    centers.row(c).transpose()
                );
            }

            double output = weights.dot(activations) + bias;
            double probability = sigmoid(output);

            int prediction = probability >= 0.5 ? 1 : 0;

            if (prediction != y(i)) {
                errors++;
            }

            double error =
                probability - static_cast<double>(y(i));

            weights -= learningRate * error * activations;
            bias -= learningRate * error;
        }

        errorsPerEpoch.push_back(errors);
    }
}

double RBF::predictProba(const Eigen::VectorXd& x) const {
    Eigen::VectorXd activations(numCenters);

    for (int c = 0; c < numCenters; c++) {
        activations(c) = activation(
            x,
            centers.row(c).transpose()
        );
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

double RBF::score(
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

std::vector<int> RBF::getErrors() const {
    return errorsPerEpoch;
}

const Eigen::MatrixXd& RBF::getCenters() const { return centers; }
const Eigen::VectorXd& RBF::getWeights() const { return weights; }
double RBF::getBias() const { return bias; }

void RBF::setParameters(
    const Eigen::MatrixXd& newCenters,
    const Eigen::VectorXd& newWeights,
    double newBias
) {
    centers = newCenters;
    weights = newWeights;
    numCenters = static_cast<int>(newCenters.rows());
    bias = newBias;
}
