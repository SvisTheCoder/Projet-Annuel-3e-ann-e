#include "svm.hpp"

SVM::SVM(double learningRate, int epochs, double lambda) {
    this->learningRate = learningRate;
    this->epochs = epochs;
    this->lambda = lambda;
    this->bias = 0.0;
}

void SVM::fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) {
    if (X.rows() == 0 || X.rows() != y.size()) {
        return;
    }

    int nFeatures = X.cols();

    weights = Eigen::VectorXd::Zero(nFeatures);
    bias = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < X.rows(); i++) {
            Eigen::VectorXd x = X.row(i).transpose();
            int label = y(i);

            double margin =
                static_cast<double>(label) * decisionFunction(x);

            if (margin >= 1.0) {
                weights -= learningRate * (2.0 * lambda * weights);
            } else {
                weights -= learningRate * (
                    2.0 * lambda * weights
                    - static_cast<double>(label) * x
                );

                bias += learningRate * static_cast<double>(label);
            }
        }
    }
}

double SVM::decisionFunction(const Eigen::VectorXd& x) const {
    return weights.dot(x) + bias;
}

int SVM::predict(const Eigen::VectorXd& x) const {
    if (decisionFunction(x) >= 0.0) {
        return 1;
    }

    return -1;
}

double SVM::score(
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

Eigen::VectorXd SVM::getWeights() const {
    return weights;
}

double SVM::getBias() const {
    return bias;
}

void SVM::setParameters(
    const Eigen::VectorXd& newWeights,
    double newBias
) {
    weights = newWeights;
    bias = newBias;
}
