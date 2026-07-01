#ifndef RBF_HPP
#define RBF_HPP

#include <Eigen/Dense>
#include <vector>

class RBF {
private:
    int numCenters;
    double sigma;
    double learningRate;
    int epochs;
    Eigen::MatrixXd centers;
    Eigen::VectorXd weights;
    double bias;
    std::vector<int> errorsPerEpoch;
    double activation(const Eigen::VectorXd& x, const Eigen::VectorXd& center) const;
    double sigmoid(double x) const;

public:
    RBF(int numCenters, double sigma, double learningRate, int epochs);
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXi& y);
    double predictProba(const Eigen::VectorXd& x) const;
    int predict(const Eigen::VectorXd& x) const;
    double score(const Eigen::MatrixXd& X, const Eigen::VectorXi& y) const;
    std::vector<int> getErrors() const;
    const Eigen::MatrixXd& getCenters() const;
    const Eigen::VectorXd& getWeights() const;
    double getBias() const;
    void setParameters(
        const Eigen::MatrixXd& newCenters,
        const Eigen::VectorXd& newWeights,
        double newBias
    );
};

#endif
