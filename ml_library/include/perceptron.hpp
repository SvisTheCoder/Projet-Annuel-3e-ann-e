#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <Eigen/Dense>
#include <vector>

class Perceptron {
private:
    double learningRate;
    int epochs;
    Eigen::VectorXd weights;
    double bias;
    std::vector<int> errorsPerEpoch;

public:
    Perceptron(double learningRate, int epochs);

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
    int predict(const Eigen::VectorXd& x) const;
    double score(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) const;

    Eigen::VectorXd getWeights() const;
    double getBias() const;
    std::vector<int> getErrors() const;
};

#endif