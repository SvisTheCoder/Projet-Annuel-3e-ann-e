#ifndef SVM_HPP
#define SVM_HPP

#include <Eigen/Dense>

class SVM {
private:
    double learningRate;
    int epochs;
    double lambda;

    Eigen::VectorXd weights;
    double bias;

public:
    SVM(double learningRate, int epochs, double lambda);

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
    int predict(const Eigen::VectorXd& x) const;
    double score(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) const;

    Eigen::VectorXd getWeights() const;
    double getBias() const;
};

#endif