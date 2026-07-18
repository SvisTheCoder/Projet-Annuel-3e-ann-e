#include "ml_api.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kImageFeatureCount = 32 * 32;

const char* model_name(MLModelType type) {
    switch (type) {
        case ML_PERCEPTRON: return "Perceptron";
        case ML_MLP: return "MLP";
        case ML_RBF: return "RBF";
        case ML_SVM: return "SVM";
        default: return "Inconnu";
    }
}

const char* score_type(MLModelType type) {
    // Lineaires = marge. MLP/RBF = meilleur score OVR.
    if (type == ML_PERCEPTRON || type == ML_SVM) {
        return "margin";
    }
    return "best_ovr_probability";
}

std::string class_name(int class_id) {
    switch (class_id) {
        case 0: return "Art déco";
        case 1: return "Art nouveau";
        case 2: return "Gothique";
        default: return "Classe " + std::to_string(class_id);
    }
}

bool load_features(
    const std::string& path,
    int expected_count,
    std::vector<double>& features
) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Impossible d'ouvrir le fichier de caractéristiques.\n";
        return false;
    }

    double value = 0.0;
    while (file >> value) {
        features.push_back(value);
    }

    if (!file.eof()) {
        std::cerr << "Le fichier de caractéristiques contient une valeur invalide.\n";
        return false;
    }
    if (static_cast<int>(features.size()) != expected_count) {
        std::cerr << "Nombre de caractéristiques invalide : "
                  << features.size() << " au lieu de " << expected_count << ".\n";
        return false;
    }
    for (double feature : features) {
        if (feature < 0.0 || feature > 1.0) {
            std::cerr << "Une caractéristique est hors de l'intervalle [0, 1].\n";
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 5) {
        std::cerr << "Usage : predict_cli chemin_modele fichier_caracteristiques "
                     "[--expected-class-count N]\n";
        return 2;
    }

    int expected_class_count = 0;
    if (argc == 5) {
        if (std::string(argv[3]) != "--expected-class-count") {
            std::cerr << "Option inconnue : " << argv[3] << '\n';
            return 2;
        }
        try {
            std::size_t parsed_characters = 0;
            expected_class_count = std::stoi(argv[4], &parsed_characters);
            if (parsed_characters != std::string(argv[4]).size()
                || expected_class_count <= 0) {
                throw std::invalid_argument("nombre invalide");
            }
        } catch (const std::exception&) {
            std::cerr << "Nombre de classes attendu invalide : " << argv[4] << '\n';
            return 2;
        }
    }

    MLModel* model = ml_load(argv[1]);
    if (model == nullptr) {
        std::cerr << "Chargement du modèle impossible : " << ml_last_error() << '\n';
        return 3;
    }

    const int feature_count = ml_feature_count(model);
    const int class_count = ml_class_count(model);
    if (expected_class_count > 0 && class_count != expected_class_count) {
        std::cerr << "Nombre de classes du modèle invalide : "
                  << class_count << " au lieu de " << expected_class_count << ".\n";
        ml_free(model);
        return 4;
    }
    if (feature_count != kImageFeatureCount) {
        std::cerr << "Le modèle attend " << feature_count
                  << " caractéristiques au lieu de 1024.\n";
        ml_free(model);
        return 4;
    }

    std::vector<double> features;
    if (!load_features(argv[2], feature_count, features)) {
        ml_free(model);
        return 4;
    }

    double score = 0.0;
    const int prediction = ml_predict_with_score(model, features.data(), &score);
    if (prediction < 0) {
        std::cerr << "Prédiction impossible : " << ml_last_error() << '\n';
        ml_free(model);
        return 5;
    }

    const MLModelType type = ml_type(model);
    // stdout = JSON seulement. Flask lit cette ligne.
    std::cout << std::setprecision(17)
              << "{\"model\":\"" << model_name(type)
              << "\",\"class_id\":" << prediction
              << ",\"class_name\":\"" << class_name(prediction)
              << "\",\"class_count\":" << class_count
              << ",\"score\":" << score
              << ",\"score_type\":\"" << score_type(type) << "\"}\n";

    ml_free(model);
    return 0;
}
