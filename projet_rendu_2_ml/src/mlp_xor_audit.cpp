#include "mlp.hpp"
#include "ml_api.h"

#include <Eigen/Dense>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr int kEpochs = 2000;
constexpr double kLearningRate = 0.1;

double accuracy(const MLP& model, const Eigen::MatrixXd& features,
                const Eigen::VectorXi& labels) {
    int correct = 0;
    for (int row = 0; row < features.rows(); ++row) {
        const int prediction = model.predict(features.row(row).transpose());
        assert(prediction == 0 || prediction == 1);
        correct += prediction == labels(row) ? 1 : 0;
    }
    return static_cast<double>(correct) / features.rows();
}

}  // namespace

int main() {
    Eigen::MatrixXd canonical_features(4, 2);
    canonical_features << 0, 0,
                          0, 1,
                          1, 0,
                          1, 1;
    Eigen::VectorXi canonical_labels(4);
    canonical_labels << 0, 1, 1, 0;
    const std::array<double, 8> canonical_flat = {0, 0, 0, 1, 1, 0, 1, 1};

    for (int label : canonical_labels) {
        assert(label == 0 || label == 1);
    }

    const std::array<unsigned int, 5> seeds = {1, 2, 3, 7, 42};
    const std::array<std::array<int, 4>, 3> orders = {{
        {{0, 1, 2, 3}},
        {{3, 2, 1, 0}},
        {{0, 3, 1, 2}},
    }};

    int perfect_runs = 0;
    int total_runs = 0;

    for (unsigned int seed : seeds) {
        for (const auto& order : orders) {
            Eigen::MatrixXd features(4, 2);
            Eigen::VectorXi labels(4);
            for (int row = 0; row < 4; ++row) {
                features.row(row) = canonical_features.row(order[row]);
                labels(row) = canonical_labels(order[row]);
            }

            std::srand(seed);
            MLP model(2, 4, kLearningRate, kEpochs);
            const Eigen::MatrixXd initial_weights = model.getInputWeights();
            assert(initial_weights.allFinite());

            model.fit(features, labels);

            const std::vector<double> losses = model.getLossHistory();
            assert(losses.size() == kEpochs);
            assert(std::isfinite(losses.front()));
            assert(std::isfinite(losses.back()));
            assert((model.getInputWeights() - initial_weights).norm() > 0.0);

            const double score = accuracy(model, canonical_features, canonical_labels);
            perfect_runs += score == 1.0 ? 1 : 0;
            ++total_runs;

            std::cout << "seed=" << seed
                      << " order=" << order[0] << order[1] << order[2] << order[3]
                      << " accuracy=" << std::fixed << std::setprecision(3) << score
                      << " loss=" << losses.front() << "->" << losses.back() << '\n';
        }
    }

    std::cout << "perfect_runs=" << perfect_runs << '/' << total_runs << '\n';
    assert(perfect_runs > 0);

    int api_perfect_runs = 0;
    int api_total_runs = 0;
    for (unsigned int seed : seeds) {
        for (const auto& order : orders) {
            std::array<double, 8> features = {};
            std::array<int, 4> labels = {};
            for (int row = 0; row < 4; ++row) {
                features[row * 2] = canonical_features(order[row], 0);
                features[row * 2 + 1] = canonical_features(order[row], 1);
                labels[row] = canonical_labels(order[row]);
            }

            MLParams params = ml_default_params();
            params.hidden_size = 6;
            params.epochs = 1500;
            params.learning_rate = 0.1;
            params.seed = seed;

            MLModel* model = ml_create(ML_MLP, 2, 2, params);
            assert(model != nullptr);
            assert(ml_train(model, features.data(), labels.data(), 4) == 1);
            const double score = ml_score(
                model, canonical_flat.data(), canonical_labels.data(), 4);
            assert(score >= 0.0 && score <= 1.0);
            api_perfect_runs += score == 1.0 ? 1 : 0;
            ++api_total_runs;

            std::cout << "api seed=" << seed
                      << " order=" << order[0] << order[1] << order[2] << order[3]
                      << " accuracy=" << std::fixed << std::setprecision(3) << score
                      << '\n';
            ml_free(model);
        }
    }

    std::cout << "api_perfect_runs=" << api_perfect_runs << '/' << api_total_runs << '\n';
    assert(api_perfect_runs > 0);
    return 0;
}
