#include "perceptron.hpp"

Perceptron::Perceptron(double learningRate, int epochs) {
    this->learningRate = learningRate;
    this->epochs = epochs;
    this->bias = 0.0;
}

void Perceptron::fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    if (X.rows() == 0 || X.rows() != y.size()) {
        return;
    }

    int nSamples = X.rows();
    int nFeatures = X.cols();

    weights = Eigen::VectorXd::Zero(nFeatures);
    bias = 0.0;
    errorsPerEpoch.clear();

    for (int epoch = 0; epoch < epochs; epoch++) {
        int errors = 0;

        for (int i = 0; i < nSamples; i++) {
            Eigen::VectorXd x = X.row(i).transpose();

            int prediction = predict(x);

            if (prediction != y(i)) {
                weights += learningRate * static_cast<double>(y(i)) * x;
                bias += learningRate * static_cast<double>(y(i));
                errors++;
            }
        }

        errorsPerEpoch.push_back(errors);
    }
}

double Perceptron::decisionFunction(const Eigen::VectorXd& x) const {
    return weights.dot(x) + bias;
}

int Perceptron::predict(const Eigen::VectorXd& x) const {
    if (decisionFunction(x) >= 0.0) {
        return 1;
    }

    return -1;
}

double Perceptron::score(
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

Eigen::VectorXd Perceptron::getWeights() const {
    return weights;
}

double Perceptron::getBias() const {
    return bias;
}

std::vector<int> Perceptron::getErrors() const {
    return errorsPerEpoch;
}

void Perceptron::setParameters(
    const Eigen::VectorXd& newWeights,
    double newBias
) {
    weights = newWeights;
    bias = newBias;
}
