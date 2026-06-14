#include "perceptron.hpp"

Perceptron::Perceptron(double learningRate, int epochs) {
    this->learningRate = learningRate;
    this->epochs = epochs;
    this->bias = 0.0;
}

void Perceptron::fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    int nSamples = X.rows();
    int nFeatures = X.cols();

    weights = Eigen::VectorXd::Zero(nFeatures);
    errorsPerEpoch.clear();

    for (int epoch = 0; epoch < epochs; epoch++) {
        int errors = 0;

        for (int i = 0; i < nSamples; i++) {
            int prediction = predict(X.row(i));

            if (prediction != y(i)) {
                weights = weights + learningRate * y(i) * X.row(i).transpose();
                bias = bias + learningRate * y(i);
                errors++;
            }
        }

        errorsPerEpoch.push_back(errors);
    }
}

int Perceptron::predict(const Eigen::VectorXd& x) const {
    double result = weights.dot(x) + bias;

    if (result >= 0) {
        return 1;
    }

    return -1;
}

double Perceptron::score(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) const {
    int correct = 0;

    for (int i = 0; i < X.rows(); i++) {
        if (predict(X.row(i)) == y(i)) {
            correct++;
        }
    }

    return (double)correct / X.rows();
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