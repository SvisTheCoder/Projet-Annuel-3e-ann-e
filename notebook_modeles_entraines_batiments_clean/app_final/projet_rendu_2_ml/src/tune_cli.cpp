#include "ml_api.h"

extern "C" {
#include "dataset.h"
}

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kSeed = 42;

struct Candidate {
    std::string name;
    MLParams params;
};

struct ActivationStats {
    double mean = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double near_zero_ratio = 0.0;
};

struct Result {
    Candidate candidate;
    double train_accuracy = 0.0;
    double validation_accuracy = 0.0;
    double validation_balanced_accuracy = 0.0;
    double duration_seconds = 0.0;
    std::vector<std::vector<int>> confusion_matrix;
    std::vector<int> prediction_distribution;
    ActivationStats activation_stats;
    bool has_activation_stats = false;
};

MLModelType parse_type(const std::string& name) {
    if (name == "perceptron") return ML_PERCEPTRON;
    if (name == "mlp") return ML_MLP;
    if (name == "rbf") return ML_RBF;
    if (name == "svm") return ML_SVM;
    return static_cast<MLModelType>(0);
}

const char* algorithm_name(MLModelType type) {
    if (type == ML_PERCEPTRON) return "Perceptron";
    if (type == ML_MLP) return "MLP";
    if (type == ML_RBF) return "RBF";
    if (type == ML_SVM) return "SVM";
    return "Unknown";
}

Candidate make_candidate(const std::string& name, int epochs, double learning_rate,
                         int hidden_size = 16, double svm_lambda = 0.001) {
    MLParams params = ml_default_params();
    params.seed = kSeed;
    params.epochs = epochs;
    params.learning_rate = learning_rate;
    params.hidden_size = hidden_size;
    params.svm_lambda = svm_lambda;
    return {name, params};
}

Candidate make_rbf_candidate(const std::string& name, int centers, double sigma,
                             int epochs = 8, double learning_rate = 0.05) {
    MLParams params = ml_default_params();
    params.seed = kSeed;
    params.epochs = epochs;
    params.learning_rate = learning_rate;
    params.rbf_centers = centers;
    params.rbf_sigma = sigma;
    return {name, params};
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            throw std::runtime_error("Liste d'entiers invalide : " + text);
        }
        const int value = std::stoi(item);
        if (value <= 0) {
            throw std::runtime_error("Les centres RBF doivent etre positifs.");
        }
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::runtime_error("Liste d'entiers vide : " + text);
    }
    return values;
}

std::vector<double> parse_double_list(const std::string& text) {
    std::vector<double> values;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            throw std::runtime_error("Liste de nombres invalide : " + text);
        }
        const double value = std::stod(item);
        if (value <= 0.0) {
            throw std::runtime_error("Les sigmas RBF doivent etre positifs.");
        }
        values.push_back(value);
    }
    if (values.empty()) {
        throw std::runtime_error("Liste de nombres vide : " + text);
    }
    return values;
}

std::string format_sigma_for_name(double sigma) {
    std::ostringstream output;
    output << std::setprecision(8) << sigma;
    std::string text = output.str();
    for (char& character : text) {
        if (character == '.') character = '_';
    }
    return text;
}

std::vector<Candidate> candidates_for(MLModelType type,
                                      const std::vector<int>& rbf_centers,
                                      const std::vector<double>& rbf_sigmas) {
    if (type == ML_PERCEPTRON) {
        return {
            make_candidate("v1_reference", 12, 0.01),
            make_candidate("more_epochs_lower_lr", 25, 0.005),
            make_candidate("more_epochs_small_lr", 30, 0.002),
            make_candidate("moderate_epochs_higher_lr", 20, 0.02),
        };
    }
    if (type == ML_SVM) {
        return {
            make_candidate("v1_reference", 12, 0.005, 16, 0.001),
            make_candidate("more_epochs_lower_lr", 25, 0.002, 16, 0.001),
            make_candidate("small_lr_low_regularization", 30, 0.001, 16, 0.0001),
            make_candidate("moderate_epochs_more_regularization", 20, 0.005, 16, 0.01),
        };
    }
    if (type == ML_RBF) {
        const std::vector<int> centers =
            rbf_centers.empty() ? std::vector<int>{16} : rbf_centers;
        const std::vector<double> sigmas =
            rbf_sigmas.empty()
                ? std::vector<double>{2.0, 4.0, 8.0, 12.0, 16.0, 24.0, 32.0}
                : rbf_sigmas;

        std::vector<Candidate> candidates;
        for (const int center_count : centers) {
            for (const double sigma : sigmas) {
                std::ostringstream name;
                name << "centers_" << center_count
                     << "_sigma_" << format_sigma_for_name(sigma);
                candidates.push_back(
                    make_rbf_candidate(name.str(), center_count, sigma)
                );
            }
        }
        return candidates;
    }
    return {
        make_candidate("v1_reference", 8, 0.03, 16),
        make_candidate("more_epochs_lower_lr", 15, 0.02, 16),
        make_candidate("more_epochs_small_lr", 20, 0.01, 16),
        make_candidate("wider_hidden_layer", 12, 0.02, 32),
    };
}

std::vector<std::vector<int>> confusion_matrix(const MLModel* model,
                                                const Dataset& dataset) {
    std::vector<std::vector<int>> matrix(
        dataset.class_count, std::vector<int>(dataset.class_count, 0));
    for (int sample = 0; sample < dataset.sample_count; ++sample) {
        const double* features = &dataset.X[sample * dataset.feature_count];
        const int prediction = ml_predict(model, features);
        if (prediction >= 0 && prediction < dataset.class_count) {
            matrix[dataset.y[sample]][prediction]++;
        }
    }
    return matrix;
}

double balanced_accuracy(const std::vector<std::vector<int>>& matrix) {
    if (matrix.empty()) return 0.0;

    // BA : moyenne du rappel de chaque classe.
    double total_recall = 0.0;
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        int class_total = 0;
        for (int value : matrix[row]) {
            class_total += value;
        }
        if (class_total > 0) {
            total_recall += static_cast<double>(matrix[row][row])
                / static_cast<double>(class_total);
        }
    }
    return total_recall / static_cast<double>(matrix.size());
}

std::vector<int> prediction_distribution(
    const std::vector<std::vector<int>>& matrix
) {
    if (matrix.empty()) return {};

    std::vector<int> distribution(matrix.front().size(), 0);
    for (const auto& row : matrix) {
        for (std::size_t column = 0; column < row.size(); ++column) {
            distribution[column] += row[column];
        }
    }
    return distribution;
}

void append_spread_indices(
    const std::vector<int>& source,
    int desired_count,
    std::vector<int>& selected
) {
    if (source.empty() || desired_count <= 0) {
        return;
    }

    for (int index = 0; index < desired_count; ++index) {
        const std::size_t source_index =
            static_cast<std::size_t>(index)
            * static_cast<std::size_t>(source.size())
            / static_cast<std::size_t>(desired_count);
        selected.push_back(
            source[std::min(source_index, source.size() - 1)]
        );
    }
}

std::vector<int> rbf_center_indices_for_class(
    const Dataset& source,
    int class_index,
    int center_count
) {
    // Centres RBF : moitie positifs, moitie negatifs, ordre stable.
    std::vector<int> positive;
    std::vector<int> negative;

    for (int sample = 0; sample < source.sample_count; ++sample) {
        if (source.y[sample] == class_index) {
            positive.push_back(sample);
        } else {
            negative.push_back(sample);
        }
    }

    std::vector<int> selected;
    selected.reserve(center_count);

    if (!positive.empty() && !negative.empty()) {
        const int positive_count = center_count / 2;
        const int negative_count = center_count - positive_count;
        append_spread_indices(positive, positive_count, selected);
        append_spread_indices(negative, negative_count, selected);
    } else {
        std::vector<int> all_samples;
        all_samples.reserve(source.sample_count);
        for (int sample = 0; sample < source.sample_count; ++sample) {
            all_samples.push_back(sample);
        }
        append_spread_indices(all_samples, center_count, selected);
    }

    return selected;
}

ActivationStats rbf_activation_stats(const Dataset& center_source,
                                     const Dataset& samples,
                                     const MLParams& params) {
    ActivationStats stats;
    if (center_source.sample_count <= 0 || samples.sample_count <= 0
        || center_source.feature_count != samples.feature_count
        || params.rbf_centers <= 0 || params.rbf_sigma <= 0.0) {
        return stats;
    }

    stats.minimum = std::numeric_limits<double>::infinity();
    stats.maximum = 0.0;
    long long value_count = 0;
    long long near_zero_count = 0;
    const double denominator = 2.0 * params.rbf_sigma * params.rbf_sigma;

    for (int class_index = 0; class_index < center_source.class_count; ++class_index) {
        const std::vector<int> center_indices = rbf_center_indices_for_class(
            center_source,
            class_index,
            params.rbf_centers
        );

        for (int sample = 0; sample < samples.sample_count; ++sample) {
            const double* x = &samples.X[sample * samples.feature_count];
            for (int center : center_indices) {
                const double* c =
                    &center_source.X[center * center_source.feature_count];
                double distance_squared = 0.0;
                for (int feature = 0; feature < samples.feature_count; ++feature) {
                    const double difference = x[feature] - c[feature];
                    distance_squared += difference * difference;
                }
                const double activation = std::exp(-distance_squared / denominator);
                stats.mean += activation;
                if (activation < stats.minimum) stats.minimum = activation;
                if (activation > stats.maximum) stats.maximum = activation;
                if (activation < 1e-12) ++near_zero_count;
                ++value_count;
            }
        }
    }

    if (value_count > 0) {
        stats.mean /= static_cast<double>(value_count);
        stats.near_zero_ratio = static_cast<double>(near_zero_count)
            / static_cast<double>(value_count);
    }
    if (!std::isfinite(stats.minimum)) stats.minimum = 0.0;
    return stats;
}

Result evaluate_candidate(MLModelType type, const Candidate& candidate,
                          const Dataset& selection_train,
                          const Dataset& validation) {
    MLModel* model = ml_create(type, selection_train.feature_count,
                               selection_train.class_count, candidate.params);
    if (model == nullptr) {
        throw std::runtime_error(std::string("Creation impossible : ") + ml_last_error());
    }

    const auto start = std::chrono::steady_clock::now();
    const int trained = ml_train(model, selection_train.X, selection_train.y,
                                 selection_train.sample_count);
    const auto end = std::chrono::steady_clock::now();
    if (!trained) {
        const std::string error = ml_last_error();
        ml_free(model);
        throw std::runtime_error("Entrainement impossible : " + error);
    }

    Result result;
    result.candidate = candidate;
    result.train_accuracy = ml_score(model, selection_train.X, selection_train.y,
                                     selection_train.sample_count);
    result.validation_accuracy = ml_score(model, validation.X, validation.y,
                                          validation.sample_count);
    result.duration_seconds = std::chrono::duration<double>(end - start).count();
    result.confusion_matrix = confusion_matrix(model, validation);
    result.validation_balanced_accuracy = balanced_accuracy(result.confusion_matrix);
    result.prediction_distribution = prediction_distribution(result.confusion_matrix);
    if (type == ML_RBF) {
        result.activation_stats = rbf_activation_stats(
            selection_train, validation, candidate.params
        );
        result.has_activation_stats = true;
    }
    ml_free(model);
    return result;
}

void write_matrix(std::ostream& output,
                  const std::vector<std::vector<int>>& matrix) {
    output << '[';
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        if (row) output << ',';
        output << '[';
        for (std::size_t column = 0; column < matrix[row].size(); ++column) {
            if (column) output << ',';
            output << matrix[row][column];
        }
        output << ']';
    }
    output << ']';
}

void write_vector(std::ostream& output, const std::vector<int>& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << values[index];
    }
    output << ']';
}

bool is_better_candidate(MLModelType type, const Result& challenger,
                         const Result& current) {
    if (type == ML_RBF) {
        if (challenger.validation_balanced_accuracy
            != current.validation_balanced_accuracy) {
            return challenger.validation_balanced_accuracy
                > current.validation_balanced_accuracy;
        }
    }
    return challenger.validation_accuracy > current.validation_accuracy;
}

void write_report(const std::filesystem::path& path, MLModelType type,
                  const Dataset& outer_train, const Dataset& outer_test,
                  const Dataset& selection_train, const Dataset& validation,
                  const std::vector<Result>& results, std::size_t selected) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream report(path);
    if (!report) throw std::runtime_error("Creation du rapport impossible");
    report << std::setprecision(17)
           << "{\n  \"algorithm\": \"" << algorithm_name(type) << "\",\n"
           << "  \"seed\": " << kSeed << ",\n"
           << "  \"outer_train_size\": " << outer_train.sample_count << ",\n"
           << "  \"outer_test_size\": " << outer_test.sample_count << ",\n"
           << "  \"selection_train_size\": " << selection_train.sample_count << ",\n"
           << "  \"validation_size\": " << validation.sample_count << ",\n"
           << "  \"outer_test_used_for_selection\": false,\n"
           << "  \"candidates\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const Result& result = results[index];
        report << "    {\"name\":\"" << result.candidate.name << "\","
               << "\"epochs\":" << result.candidate.params.epochs << ','
               << "\"learning_rate\":" << result.candidate.params.learning_rate << ','
               << "\"hidden_size\":" << result.candidate.params.hidden_size << ','
               << "\"rbf_centers\":" << result.candidate.params.rbf_centers << ','
               << "\"rbf_sigma\":" << result.candidate.params.rbf_sigma << ','
               << "\"svm_lambda\":" << result.candidate.params.svm_lambda << ','
               << "\"train_accuracy\":" << result.train_accuracy << ','
               << "\"validation_accuracy\":" << result.validation_accuracy << ','
               << "\"validation_balanced_accuracy\":"
               << result.validation_balanced_accuracy << ','
               << "\"duration_seconds\":" << result.duration_seconds << ','
               << "\"validation_prediction_distribution\":";
        write_vector(report, result.prediction_distribution);
        report << ',';
        if (result.has_activation_stats) {
            report << "\"activation_mean\":" << result.activation_stats.mean << ','
                   << "\"activation_min\":" << result.activation_stats.minimum << ','
                   << "\"activation_max\":" << result.activation_stats.maximum << ','
                   << "\"activation_near_zero_ratio\":"
                   << result.activation_stats.near_zero_ratio << ',';
        }
        report << "\"validation_confusion_matrix\":";
        write_matrix(report, result.confusion_matrix);
        report << '}' << (index + 1 == results.size() ? "\n" : ",\n");
    }
    report << "  ],\n  \"selected_candidate\": \""
           << results[selected].candidate.name << "\"\n}\n";
}

void print_usage() {
    std::cerr
        << "Usage : tune_cli --model perceptron|mlp|rbf|svm --csv fichier.csv "
        << "--report rapport.json [--rbf-centers 16,32] "
        << "[--rbf-sigmas 2,4,8]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if ((argc - 1) % 2 != 0) {
        print_usage();
        return 2;
    }

    std::string model_name;
    std::string csv_path;
    std::string report_path;
    std::vector<int> rbf_centers;
    std::vector<double> rbf_sigmas;

    try {
        for (int index = 1; index + 1 < argc; index += 2) {
            const std::string option = argv[index];
            if (option == "--model") {
                model_name = argv[index + 1];
            } else if (option == "--csv") {
                csv_path = argv[index + 1];
            } else if (option == "--report") {
                report_path = argv[index + 1];
            } else if (option == "--rbf-centers") {
                rbf_centers = parse_int_list(argv[index + 1]);
            } else if (option == "--rbf-sigmas") {
                rbf_sigmas = parse_double_list(argv[index + 1]);
            } else {
                std::cerr << "Option inconnue : " << option << '\n';
                return 2;
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }

    const MLModelType type = parse_type(model_name);
    if (type == 0 || csv_path.empty() || report_path.empty()) {
        print_usage();
        return 2;
    }

    Dataset dataset = {};
    Dataset outer_train = {};
    Dataset outer_test = {};
    Dataset selection_train = {};
    Dataset validation = {};
    // Split ext. : outer_test reste bloque pendant le tuning.
    if (!dataset_load_csv(csv_path.c_str(), &dataset)
        || dataset.feature_count != 1024 || dataset.class_count != 3
        || !dataset_split(&dataset, 0.20, kSeed, &outer_train, &outer_test)
        || !dataset_split(&outer_train, 0.20, kSeed,
                          &selection_train, &validation)) {
        std::cerr << "Chargement ou separation du dataset impossible.\n";
        dataset_free(&dataset);
        dataset_free(&outer_train);
        dataset_free(&outer_test);
        dataset_free(&selection_train);
        dataset_free(&validation);
        return 3;
    }

    try {
        const auto candidates = candidates_for(type, rbf_centers, rbf_sigmas);
        std::vector<Result> results;
        std::size_t selected = 0;
        for (const Candidate& candidate : candidates) {
            Result result = evaluate_candidate(type, candidate,
                                               selection_train, validation);
            std::cout << algorithm_name(type) << " / " << candidate.name
                      << " : train=" << std::fixed << std::setprecision(4)
                      << result.train_accuracy
                      << ", validation=" << result.validation_accuracy
                      << ", balanced=" << result.validation_balanced_accuracy
                      << ", duree=" << result.duration_seconds << " s\n";
            results.push_back(result);
            // Choix sur validation uniquement. Jamais sur outer_test.
            if (is_better_candidate(type, results.back(), results[selected])) {
                selected = results.size() - 1;
            }
        }
        write_report(report_path, type, outer_train, outer_test,
                     selection_train, validation, results, selected);
        std::cout << "Candidat retenu sur validation : "
                  << results[selected].candidate.name << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        dataset_free(&dataset);
        dataset_free(&outer_train);
        dataset_free(&outer_test);
        dataset_free(&selection_train);
        dataset_free(&validation);
        return 4;
    }

    dataset_free(&dataset);
    dataset_free(&outer_train);
    dataset_free(&outer_test);
    dataset_free(&selection_train);
    dataset_free(&validation);
    return 0;
}
