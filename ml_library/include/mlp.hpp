#ifndef MLP_HPP
#define MLP_HPP

#include <Eigen/Dense>
#include <vector>

class MLP {
private:
    int inputSize;
    int hiddenSize;
    double learningRate;
    int epochs;

    Eigen::MatrixXd W1;
    Eigen::VectorXd b1;
    Eigen::VectorXd W2;
    double b2;

    std::vector<double> lossHistory;

    double sigmoid(double x) const;
    double sigmoidDerivative(double value) const;

public:
    MLP(int inputSize, int hiddenSize, double learningRate, int epochs);

    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
    double predictProba(const Eigen::VectorXd& x) const;
    int predict(const Eigen::VectorXd& x) const;
    double score(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) const;

    std::vector<double> getLossHistory() const;
    const Eigen::MatrixXd& getInputWeights() const;
    const Eigen::VectorXd& getHiddenBias() const;
    const Eigen::VectorXd& getOutputWeights() const;
    double getOutputBias() const;
    void setParameters(
        const Eigen::MatrixXd& newW1,
        const Eigen::VectorXd& newB1,
        const Eigen::VectorXd& newW2,
        double newB2
    );
};

#endif
