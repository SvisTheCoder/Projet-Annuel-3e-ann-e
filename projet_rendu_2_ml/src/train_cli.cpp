#include "ml_api.h"

extern "C" {
#include "dataset.h"
}

#ifdef _WIN32
#include <windows.h>
#include <bcrypt.h>
#endif

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::string model_name;
    std::string csv_path;
    std::string output_path;
    std::string report_path;
    MLParams params = ml_default_params();
};

void print_usage() {
    std::cerr
        << "Usage : train_cli --model perceptron|mlp|rbf|svm "
        << "--csv fichier.csv --output modele.model --report rapport.json [options]\n"
        << "Options : --epochs N --learning-rate X --hidden-size N "
        << "--rbf-centers N --rbf-sigma X --svm-lambda X --seed N\n";
}

bool read_value(int& index, int argc, char** argv, std::string& value) {
    if (index + 1 >= argc) return false;
    value = argv[++index];
    return true;
}

bool parse_arguments(int argc, char** argv, Arguments& arguments) {
    for (int index = 1; index < argc; ++index) {
        const std::string option = argv[index];
        std::string value;
        if (!read_value(index, argc, argv, value)) return false;

        try {
            if (option == "--model") arguments.model_name = value;
            else if (option == "--csv") arguments.csv_path = value;
            else if (option == "--output") arguments.output_path = value;
            else if (option == "--report") arguments.report_path = value;
            else if (option == "--epochs") arguments.params.epochs = std::stoi(value);
            else if (option == "--learning-rate") arguments.params.learning_rate = std::stod(value);
            else if (option == "--hidden-size") arguments.params.hidden_size = std::stoi(value);
            else if (option == "--rbf-centers") arguments.params.rbf_centers = std::stoi(value);
            else if (option == "--rbf-sigma") arguments.params.rbf_sigma = std::stod(value);
            else if (option == "--svm-lambda") arguments.params.svm_lambda = std::stod(value);
            else if (option == "--seed") arguments.params.seed = static_cast<unsigned int>(std::stoul(value));
            else return false;
        } catch (const std::exception&) {
            return false;
        }
    }
    return !arguments.model_name.empty()
        && !arguments.csv_path.empty()
        && !arguments.output_path.empty()
        && !arguments.report_path.empty();
}

MLModelType parse_model_type(const std::string& name) {
    if (name == "perceptron") return ML_PERCEPTRON;
    if (name == "mlp") return ML_MLP;
    if (name == "rbf") return ML_RBF;
    if (name == "svm") return ML_SVM;
    return static_cast<MLModelType>(0);
}

const char* algorithm_name(MLModelType type) {
    switch (type) {
        case ML_PERCEPTRON: return "Perceptron";
        case ML_MLP: return "MLP";
        case ML_RBF: return "RBF";
        case ML_SVM: return "SVM";
        default: return "Unknown";
    }
}

const char* score_type(MLModelType type) {
    return type == ML_PERCEPTRON || type == ML_SVM
        ? "margin"
        : "best_ovr_probability";
}

void apply_simple_defaults(Arguments& arguments, MLModelType type) {
    if (type == ML_PERCEPTRON) {
        arguments.params.epochs = 12;
        arguments.params.learning_rate = 0.01;
    } else if (type == ML_MLP) {
        arguments.params.epochs = 8;
        arguments.params.learning_rate = 0.03;
        arguments.params.hidden_size = 16;
    } else if (type == ML_RBF) {
        arguments.params.epochs = 8;
        arguments.params.learning_rate = 0.05;
        arguments.params.rbf_centers = 16;
        arguments.params.rbf_sigma = 1.0;
    } else if (type == ML_SVM) {
        arguments.params.epochs = 12;
        arguments.params.learning_rate = 0.005;
        arguments.params.svm_lambda = 0.001;
    }
}

bool option_was_provided(int argc, char** argv, const std::string& option) {
    for (int index = 1; index < argc; ++index) {
        if (argv[index] == option) return true;
    }
    return false;
}

bool validate_dataset(const Dataset& dataset, std::map<int, int>& distribution) {
    for (int sample = 0; sample < dataset.sample_count; ++sample) {
        distribution[dataset.y[sample]]++;
    }

    std::cout << "Nombre de lignes : " << dataset.sample_count << '\n'
              << "Nombre de colonnes : " << dataset.feature_count + 1 << '\n'
              << "Nombre de features : " << dataset.feature_count << '\n'
              << "Labels présents :";
    for (const auto& entry : distribution) std::cout << ' ' << entry.first;
    std::cout << '\n'
              << "Art déco (0) : " << distribution[0] << '\n'
              << "Art nouveau (1) : " << distribution[1] << '\n'
              << "Gothique (2) : " << distribution[2] << '\n';

    return dataset.feature_count == 1024
        && distribution.size() == 3
        && distribution[0] > 0
        && distribution[1] > 0
        && distribution[2] > 0;
}

#ifdef _WIN32
std::string sha256_file(const std::filesystem::path& path) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash_handle = nullptr;
    DWORD object_size = 0;
    DWORD hash_size = 0;
    DWORD result_size = 0;

    if (BCryptOpenAlgorithmProvider(
            &algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0
        || BCryptGetProperty(
            algorithm, BCRYPT_OBJECT_LENGTH,
            reinterpret_cast<PUCHAR>(&object_size), sizeof(object_size),
            &result_size, 0) != 0
        || BCryptGetProperty(
            algorithm, BCRYPT_HASH_LENGTH,
            reinterpret_cast<PUCHAR>(&hash_size), sizeof(hash_size),
            &result_size, 0) != 0) {
        if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
        throw std::runtime_error("Initialisation SHA-256 impossible");
    }

    std::vector<UCHAR> object_buffer(object_size);
    std::vector<UCHAR> digest(hash_size);
    if (BCryptCreateHash(
            algorithm, &hash_handle, object_buffer.data(), object_size,
            nullptr, 0, 0) != 0) {
        BCryptCloseAlgorithmProvider(algorithm, 0);
        throw std::runtime_error("Création SHA-256 impossible");
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        BCryptDestroyHash(hash_handle);
        BCryptCloseAlgorithmProvider(algorithm, 0);
        throw std::runtime_error("Ouverture du CSV impossible pour SHA-256");
    }

    std::array<char, 65536> buffer{};
    while (file) {
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto bytes_read = file.gcount();
        if (bytes_read > 0 && BCryptHashData(
                hash_handle,
                reinterpret_cast<PUCHAR>(buffer.data()),
                static_cast<ULONG>(bytes_read), 0) != 0) {
            BCryptDestroyHash(hash_handle);
            BCryptCloseAlgorithmProvider(algorithm, 0);
            throw std::runtime_error("Calcul SHA-256 impossible");
        }
    }

    const bool success = BCryptFinishHash(
        hash_handle, digest.data(), hash_size, 0) == 0;
    BCryptDestroyHash(hash_handle);
    BCryptCloseAlgorithmProvider(algorithm, 0);
    if (!success) throw std::runtime_error("Finalisation SHA-256 impossible");

    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (UCHAR byte : digest) output << std::setw(2) << static_cast<int>(byte);
    return output.str();
}
#else
std::string sha256_file(const std::filesystem::path&) {
    throw std::runtime_error("SHA-256 non implémenté sur cette plateforme");
}
#endif

std::vector<std::vector<int>> confusion_matrix(
    const MLModel* model,
    const Dataset& dataset
) {
    std::vector<std::vector<int>> matrix(
        dataset.class_count,
        std::vector<int>(dataset.class_count, 0)
    );
    for (int sample = 0; sample < dataset.sample_count; ++sample) {
        const double* features = &dataset.X[sample * dataset.feature_count];
        const int predicted = ml_predict(model, features);
        if (predicted >= 0 && predicted < dataset.class_count) {
            matrix[dataset.y[sample]][predicted]++;
        }
    }
    return matrix;
}

bool same_predictions(
    const MLModel* first,
    const MLModel* second,
    const Dataset& dataset
) {
    for (int sample = 0; sample < dataset.sample_count; ++sample) {
        const double* features = &dataset.X[sample * dataset.feature_count];
        if (ml_predict(first, features) != ml_predict(second, features)) return false;
    }
    return true;
}

std::string json_escape(const std::string& value) {
    std::ostringstream output;
    for (unsigned char character : value) {
        switch (character) {
            case '\\': output << "\\\\"; break;
            case '"': output << "\\\""; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default: output << character; break;
        }
    }
    return output.str();
}

std::string utc_now() {
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

void write_report(
    const Arguments& arguments,
    MLModelType type,
    const std::string& csv_sha256,
    int train_size,
    int test_size,
    double duration,
    double train_accuracy,
    double test_accuracy,
    double reload_accuracy,
    const std::vector<std::vector<int>>& matrix,
    std::uintmax_t model_size
) {
    const std::filesystem::path report_path(arguments.report_path);
    if (!report_path.parent_path().empty()) {
        std::filesystem::create_directories(report_path.parent_path());
    }
    std::ofstream report(report_path);
    if (!report) throw std::runtime_error("Création du rapport JSON impossible");

    report << std::setprecision(17)
           << "{\n"
           << "  \"algorithm\": \"" << algorithm_name(type) << "\",\n"
           << "  \"model_file\": \""
           << json_escape(std::filesystem::path(arguments.output_path).filename().string()) << "\",\n"
           << "  \"csv_file\": \""
           << json_escape(std::filesystem::path(arguments.csv_path).filename().string()) << "\",\n"
           << "  \"csv_sha256\": \"" << csv_sha256 << "\",\n"
           << "  \"feature_count\": 1024,\n"
           << "  \"class_count\": 3,\n"
           << "  \"classes\": {\"0\": \"Art déco\", \"1\": \"Art nouveau\", \"2\": \"Gothique\"},\n"
           << "  \"train_size\": " << train_size << ",\n"
           << "  \"test_size\": " << test_size << ",\n"
           << "  \"seed\": " << arguments.params.seed << ",\n"
           << "  \"parameters\": {\n"
           << "    \"epochs\": " << arguments.params.epochs << ",\n"
           << "    \"learning_rate\": " << arguments.params.learning_rate << ",\n"
           << "    \"hidden_size\": " << arguments.params.hidden_size << ",\n"
           << "    \"rbf_centers\": " << arguments.params.rbf_centers << ",\n"
           << "    \"rbf_sigma\": " << arguments.params.rbf_sigma << ",\n"
           << "    \"svm_lambda\": " << arguments.params.svm_lambda << "\n"
           << "  },\n"
           << "  \"duration_seconds\": " << duration << ",\n"
           << "  \"train_accuracy\": " << train_accuracy << ",\n"
           << "  \"test_accuracy\": " << test_accuracy << ",\n"
           << "  \"confusion_matrix\": [\n";
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        report << "    [";
        for (std::size_t column = 0; column < matrix[row].size(); ++column) {
            if (column) report << ", ";
            report << matrix[row][column];
        }
        report << "]" << (row + 1 == matrix.size() ? "\n" : ",\n");
    }
    report << "  ],\n"
           << "  \"score_type\": \"" << score_type(type) << "\",\n"
           << "  \"reload_validation\": true,\n"
           << "  \"reload_accuracy\": " << reload_accuracy << ",\n"
           << "  \"model_size_bytes\": " << model_size << ",\n"
           << "  \"created_at\": \"" << utc_now() << "\"\n"
           << "}\n";
    if (!report) throw std::runtime_error("Écriture du rapport JSON impossible");
}

void free_datasets(Dataset& dataset, Dataset& train, Dataset& test) {
    dataset_free(&dataset);
    dataset_free(&train);
    dataset_free(&test);
}

}  // namespace

int main(int argc, char** argv) {
    Arguments arguments;
    if (!parse_arguments(argc, argv, arguments)) {
        print_usage();
        return 2;
    }

    const MLModelType type = parse_model_type(arguments.model_name);
    if (type == 0) {
        std::cerr << "Type de modèle inconnu : " << arguments.model_name << '\n';
        return 2;
    }

    const MLParams parsed_params = arguments.params;
    apply_simple_defaults(arguments, type);
    if (option_was_provided(argc, argv, "--epochs")) arguments.params.epochs = parsed_params.epochs;
    if (option_was_provided(argc, argv, "--learning-rate")) arguments.params.learning_rate = parsed_params.learning_rate;
    if (option_was_provided(argc, argv, "--hidden-size")) arguments.params.hidden_size = parsed_params.hidden_size;
    if (option_was_provided(argc, argv, "--rbf-centers")) arguments.params.rbf_centers = parsed_params.rbf_centers;
    if (option_was_provided(argc, argv, "--rbf-sigma")) arguments.params.rbf_sigma = parsed_params.rbf_sigma;
    if (option_was_provided(argc, argv, "--svm-lambda")) arguments.params.svm_lambda = parsed_params.svm_lambda;
    if (option_was_provided(argc, argv, "--seed")) arguments.params.seed = parsed_params.seed;

    Dataset dataset = {};
    Dataset train = {};
    Dataset test = {};
    std::map<int, int> distribution;
    if (!dataset_load_csv(arguments.csv_path.c_str(), &dataset)) {
        std::cerr << "Impossible de charger le CSV : " << arguments.csv_path << '\n';
        return 3;
    }
    if (!validate_dataset(dataset, distribution)) {
        std::cerr << "Le CSV doit contenir 1024 features et exactement les labels 0, 1, 2.\n";
        free_datasets(dataset, train, test);
        return 3;
    }
    if (!dataset_split(&dataset, 0.20, arguments.params.seed, &train, &test)) {
        std::cerr << "Impossible de séparer le dataset.\n";
        free_datasets(dataset, train, test);
        return 3;
    }

    std::string csv_sha256;
    try {
        csv_sha256 = sha256_file(arguments.csv_path);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        free_datasets(dataset, train, test);
        return 3;
    }

    std::cout << "Train size : " << train.sample_count << '\n'
              << "Test size : " << test.sample_count << '\n'
              << "CSV SHA-256 : " << csv_sha256 << '\n'
              << "epochs=" << arguments.params.epochs
              << ", learning_rate=" << arguments.params.learning_rate
              << ", hidden_size=" << arguments.params.hidden_size
              << ", rbf_centers=" << arguments.params.rbf_centers
              << ", rbf_sigma=" << arguments.params.rbf_sigma
              << ", svm_lambda=" << arguments.params.svm_lambda
              << ", seed=" << arguments.params.seed << '\n';

    MLModel* model = ml_create(type, train.feature_count, train.class_count, arguments.params);
    if (model == nullptr) {
        std::cerr << "Création du modèle impossible : " << ml_last_error() << '\n';
        free_datasets(dataset, train, test);
        return 4;
    }

    const auto start = std::chrono::steady_clock::now();
    const int trained = ml_train(model, train.X, train.y, train.sample_count);
    const auto end = std::chrono::steady_clock::now();
    if (!trained) {
        std::cerr << "Entraînement impossible : " << ml_last_error() << '\n';
        ml_free(model);
        free_datasets(dataset, train, test);
        return 4;
    }

    const double train_accuracy = ml_score(model, train.X, train.y, train.sample_count);
    const double test_accuracy = ml_score(model, test.X, test.y, test.sample_count);
    const double duration = std::chrono::duration<double>(end - start).count();
    const auto matrix = confusion_matrix(model, test);

    const std::filesystem::path output_path(arguments.output_path);
    const std::filesystem::path temporary_path = output_path.string() + ".validation.tmp";
    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::filesystem::remove(temporary_path);
    if (!ml_save(model, temporary_path.string().c_str())) {
        std::cerr << "Sauvegarde temporaire impossible : " << ml_last_error() << '\n';
        ml_free(model);
        free_datasets(dataset, train, test);
        return 5;
    }

    MLModel* reloaded = ml_load(temporary_path.string().c_str());
    const bool reload_valid = reloaded != nullptr
        && ml_feature_count(reloaded) == train.feature_count
        && ml_class_count(reloaded) == train.class_count
        && same_predictions(model, reloaded, test);
    if (!reload_valid) {
        std::cerr << "Validation après rechargement impossible : " << ml_last_error() << '\n';
        if (reloaded) ml_free(reloaded);
        std::filesystem::remove(temporary_path);
        ml_free(model);
        free_datasets(dataset, train, test);
        return 5;
    }

    const double reload_accuracy = ml_score(reloaded, test.X, test.y, test.sample_count);
    ml_free(reloaded);
    std::filesystem::remove(output_path);
    std::filesystem::rename(temporary_path, output_path);
    const auto model_size = std::filesystem::file_size(output_path);

    try {
        write_report(
            arguments, type, csv_sha256,
            train.sample_count, test.sample_count, duration,
            train_accuracy, test_accuracy, reload_accuracy,
            matrix, model_size
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        ml_free(model);
        free_datasets(dataset, train, test);
        return 6;
    }

    std::cout << std::fixed << std::setprecision(4)
              << "Train accuracy : " << train_accuracy << '\n'
              << "Test accuracy : " << test_accuracy << '\n'
              << "Reload accuracy : " << reload_accuracy << '\n'
              << "Durée entraînement (s) : " << duration << '\n'
              << "Modèle sauvegardé : " << output_path << '\n'
              << "Rapport JSON : " << arguments.report_path << '\n';

    ml_free(model);
    free_datasets(dataset, train, test);
    return 0;
}
