#include "ml_api.h"

extern "C" {
#include "dataset.h"
}

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned int kSeed = 42;

struct Candidate {
    std::string name;
    MLParams params;
};

struct Result {
    Candidate candidate;
    double train_accuracy = 0.0;
    double validation_accuracy = 0.0;
    double duration_seconds = 0.0;
    std::vector<std::vector<int>> confusion_matrix;
};

MLModelType parse_type(const std::string& name) {
    if (name == "perceptron") return ML_PERCEPTRON;
    if (name == "mlp") return ML_MLP;
    if (name == "svm") return ML_SVM;
    return static_cast<MLModelType>(0);
}

const char* algorithm_name(MLModelType type) {
    if (type == ML_PERCEPTRON) return "Perceptron";
    if (type == ML_MLP) return "MLP";
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

std::vector<Candidate> candidates_for(MLModelType type) {
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

Result evaluate_candidate(MLModelType type, const Candidate& candidate,
                          const Dataset& selection_train,
                          const Dataset& validation) {
    MLModel* model = ml_create(type, selection_train.feature_count,
                               selection_train.class_count, candidate.params);
    if (model == nullptr) {
        throw std::runtime_error(std::string("Création impossible : ") + ml_last_error());
    }

    const auto start = std::chrono::steady_clock::now();
    const int trained = ml_train(model, selection_train.X, selection_train.y,
                                 selection_train.sample_count);
    const auto end = std::chrono::steady_clock::now();
    if (!trained) {
        const std::string error = ml_last_error();
        ml_free(model);
        throw std::runtime_error("Entraînement impossible : " + error);
    }

    Result result;
    result.candidate = candidate;
    result.train_accuracy = ml_score(model, selection_train.X, selection_train.y,
                                     selection_train.sample_count);
    result.validation_accuracy = ml_score(model, validation.X, validation.y,
                                          validation.sample_count);
    result.duration_seconds = std::chrono::duration<double>(end - start).count();
    result.confusion_matrix = confusion_matrix(model, validation);
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

void write_report(const std::filesystem::path& path, MLModelType type,
                  const Dataset& outer_train, const Dataset& outer_test,
                  const Dataset& selection_train, const Dataset& validation,
                  const std::vector<Result>& results, std::size_t selected) {
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }
    std::ofstream report(path);
    if (!report) throw std::runtime_error("Création du rapport impossible");
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
               << "\"svm_lambda\":" << result.candidate.params.svm_lambda << ','
               << "\"train_accuracy\":" << result.train_accuracy << ','
               << "\"validation_accuracy\":" << result.validation_accuracy << ','
               << "\"duration_seconds\":" << result.duration_seconds << ','
               << "\"validation_confusion_matrix\":";
        write_matrix(report, result.confusion_matrix);
        report << '}' << (index + 1 == results.size() ? "\n" : ",\n");
    }
    report << "  ],\n  \"selected_candidate\": \""
           << results[selected].candidate.name << "\"\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_name;
    std::string csv_path;
    std::string report_path;
    for (int index = 1; index + 1 < argc; index += 2) {
        const std::string option = argv[index];
        if (option == "--model") model_name = argv[index + 1];
        else if (option == "--csv") csv_path = argv[index + 1];
        else if (option == "--report") report_path = argv[index + 1];
        else {
            std::cerr << "Option inconnue : " << option << '\n';
            return 2;
        }
    }
    const MLModelType type = parse_type(model_name);
    if (type == 0 || csv_path.empty() || report_path.empty()) {
        std::cerr << "Usage : tune_cli --model perceptron|mlp|svm --csv fichier.csv "
                     "--report rapport.json\n";
        return 2;
    }

    Dataset dataset = {};
    Dataset outer_train = {};
    Dataset outer_test = {};
    Dataset selection_train = {};
    Dataset validation = {};
    if (!dataset_load_csv(csv_path.c_str(), &dataset)
        || dataset.feature_count != 1024 || dataset.class_count != 3
        || !dataset_split(&dataset, 0.20, kSeed, &outer_train, &outer_test)
        || !dataset_split(&outer_train, 0.20, kSeed,
                          &selection_train, &validation)) {
        std::cerr << "Chargement ou séparation du dataset impossible.\n";
        dataset_free(&dataset);
        dataset_free(&outer_train);
        dataset_free(&outer_test);
        dataset_free(&selection_train);
        dataset_free(&validation);
        return 3;
    }

    try {
        const auto candidates = candidates_for(type);
        std::vector<Result> results;
        std::size_t selected = 0;
        for (const Candidate& candidate : candidates) {
            Result result = evaluate_candidate(type, candidate,
                                               selection_train, validation);
            std::cout << algorithm_name(type) << " / " << candidate.name
                      << " : train=" << std::fixed << std::setprecision(4)
                      << result.train_accuracy
                      << ", validation=" << result.validation_accuracy
                      << ", durée=" << result.duration_seconds << " s\n";
            results.push_back(result);
            if (results.back().validation_accuracy
                > results[selected].validation_accuracy) {
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
