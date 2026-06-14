#include "svm.hpp"

SVM::SVM(double learningRate, int epochs, double lambda) {
    this->learningRate = learningRate;
    this->epochs = epochs;
    this->lambda = lambda;
    this->bias = 0.0;
}

void SVM::fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    int nFeatures = X.cols();
    weights = Eigen::VectorXd::Zero(nFeatures);

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < X.rows(); i++) {
            Eigen::VectorXd x = X.row(i);
            int label = y(i);

            double condition = label * (weights.dot(x) + bias);

            if (condition >= 1) {
                weights = weights - learningRate * (2 * lambda * weights);
            } else {
                weights = weights - learningRate * (2 * lambda * weights - label * x);
                bias = bias + learningRate * label;
            }
        }
    }
}

int SVM::predict(const Eigen::VectorXd& x) const {
    double result = weights.dot(x) + bias;

    if (result >= 0) {
        return 1;
    }

    return -1;
}

double SVM::score(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) const {
    int correct = 0;

    for (int i = 0; i < X.rows(); i++) {
        if (predict(X.row(i)) == y(i)) {
            correct++;
        }
    }

    return (double)correct / X.rows();
}

Eigen::VectorXd SVM::getWeights() const {
    return weights;
}

double SVM::getBias() const {
    return bias;
}