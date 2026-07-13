#include "rbf.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

void append_spread_indices(
    const std::vector<int>& source,
    int desiredCount,
    std::vector<int>& selected
) {
    if (source.empty() || desiredCount <= 0) {
        return;
    }

    for (int index = 0; index < desiredCount; index++) {
        const std::size_t sourceIndex =
            static_cast<std::size_t>(index)
            * static_cast<std::size_t>(source.size())
            / static_cast<std::size_t>(desiredCount);
        selected.push_back(
            source[std::min(sourceIndex, source.size() - 1)]
        );
    }
}

std::vector<int> select_center_indices(
    const Eigen::VectorXi& y,
    int sampleCount,
    int centerCount
) {
    std::vector<int> positive;
    std::vector<int> negative;

    for (int sample = 0; sample < sampleCount; sample++) {
        if (y(sample) == 1) {
            positive.push_back(sample);
        } else {
            negative.push_back(sample);
        }
    }

    std::vector<int> selected;
    selected.reserve(centerCount);

    if (!positive.empty() && !negative.empty()) {
        const int positiveCount = centerCount / 2;
        const int negativeCount = centerCount - positiveCount;

        append_spread_indices(positive, positiveCount, selected);
        append_spread_indices(negative, negativeCount, selected);
    } else {
        std::vector<int> allSamples;
        allSamples.reserve(sampleCount);

        for (int sample = 0; sample < sampleCount; sample++) {
            allSamples.push_back(sample);
        }

        append_spread_indices(allSamples, centerCount, selected);
    }

    return selected;
}

}  // namespace

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

    const std::vector<int> centerIndices = select_center_indices(
        y,
        static_cast<int>(X.rows()),
        numCenters
    );

    for (int c = 0; c < numCenters; c++) {
        centers.row(c) = X.row(centerIndices[c % centerIndices.size()]);
    }

    Eigen::MatrixXd activationMatrix(X.rows(), numCenters);

    for (int i = 0; i < X.rows(); i++) {
        for (int c = 0; c < numCenters; c++) {
            double distanceSquared = 0.0;

            for (int feature = 0; feature < inputSize; feature++) {
                const double difference =
                    X(i, feature) - centers(c, feature);
                distanceSquared += difference * difference;
            }

            activationMatrix(i, c) = std::exp(
                -distanceSquared / (2.0 * sigma * sigma)
            );
        }
    }

    errorsPerEpoch.clear();

    for (int epoch = 0; epoch < epochs; epoch++) {
        int errors = 0;

        for (int i = 0; i < X.rows(); i++) {
            double output = bias;

            for (int c = 0; c < numCenters; c++) {
                output += weights(c) * activationMatrix(i, c);
            }

            double probability = sigmoid(output);

            int prediction = probability >= 0.5 ? 1 : 0;

            if (prediction != y(i)) {
                errors++;
            }

            double error =
                probability - static_cast<double>(y(i));

            for (int c = 0; c < numCenters; c++) {
                weights(c) -= learningRate * error * activationMatrix(i, c);
            }
            bias -= learningRate * error;
        }

        errorsPerEpoch.push_back(errors);
    }
}

double RBF::predictProba(const Eigen::VectorXd& x) const {
    Eigen::VectorXd activations(numCenters);

    for (int c = 0; c < numCenters; c++) {
        double distanceSquared = 0.0;

        for (int feature = 0; feature < x.size(); feature++) {
            const double difference = x(feature) - centers(c, feature);
            distanceSquared += difference * difference;
        }

        activations(c) = std::exp(
            -distanceSquared / (2.0 * sigma * sigma)
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
