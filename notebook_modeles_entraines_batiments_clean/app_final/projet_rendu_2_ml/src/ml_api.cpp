#include "ml_api.h"

#include "mlp.hpp"
#include "perceptron.hpp"
#include "rbf.hpp"
#include "svm.hpp"

#include <Eigen/Dense>

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

static std::string g_last_error;

static void set_error(const std::string& message) {
    g_last_error = message;
}

struct MLModel {
    MLModelType type;
    int feature_count;
    int class_count;
    MLParams params;
    std::vector<std::unique_ptr<Perceptron>> perceptrons;
    std::vector<std::unique_ptr<MLP>> mlps;
    std::vector<std::unique_ptr<RBF>> rbfs;
    std::vector<std::unique_ptr<SVM>> svms;
};

MLParams ml_default_params(void) {
    MLParams params = {};
    params.epochs = 100;
    params.learning_rate = 0.05;
    params.hidden_size = 16;
    params.rbf_centers = 12;
    params.rbf_sigma = 1.0;
    params.svm_lambda = 0.001;
    params.seed = 42;
    return params;
}

static bool valid_type(MLModelType type) {
    return type >= ML_PERCEPTRON && type <= ML_SVM;
}

static void create_binary_models(MLModel& model) {
    std::srand(model.params.seed);
    for (int class_index = 0; class_index < model.class_count; ++class_index) {
        switch (model.type) {
            case ML_PERCEPTRON:
                model.perceptrons.push_back(std::make_unique<Perceptron>(
                    model.params.learning_rate, model.params.epochs));
                break;
            case ML_MLP:
                model.mlps.push_back(std::make_unique<MLP>(
                    model.feature_count, model.params.hidden_size,
                    model.params.learning_rate, model.params.epochs));
                break;
            case ML_RBF:
                model.rbfs.push_back(std::make_unique<RBF>(
                    model.params.rbf_centers, model.params.rbf_sigma,
                    model.params.learning_rate, model.params.epochs));
                break;
            case ML_SVM:
                model.svms.push_back(std::make_unique<SVM>(
                    model.params.learning_rate, model.params.epochs,
                    model.params.svm_lambda));
                break;
        }
    }
}

MLModel* ml_create(
    MLModelType type,
    int feature_count,
    int class_count,
    MLParams params
) {
    g_last_error.clear();
    if (!valid_type(type)) {
        set_error("Type de modele inconnu");
        return nullptr;
    }
    if (feature_count <= 0 || class_count < 2) {
        set_error("Dimensions invalides");
        return nullptr;
    }
    if (params.epochs <= 0 || params.learning_rate <= 0.0) {
        set_error("Parametres d'apprentissage invalides");
        return nullptr;
    }
    if (params.hidden_size <= 0) params.hidden_size = 1;
    if (params.rbf_centers <= 0) params.rbf_centers = 1;
    if (params.rbf_sigma <= 0.0) {
        set_error("Sigma RBF doit etre positif");
        return nullptr;
    }

    try {
        auto model = std::make_unique<MLModel>();
        model->type = type;
        model->feature_count = feature_count;
        model->class_count = class_count;
        model->params = params;
        create_binary_models(*model);
        return model.release();
    } catch (const std::exception& exception) {
        set_error(exception.what());
        return nullptr;
    }
}

static Eigen::MatrixXd copy_features(
    const double* X,
    int sample_count,
    int feature_count
) {
    Eigen::MatrixXd result(sample_count, feature_count);
    for (int sample = 0; sample < sample_count; ++sample) {
        for (int feature = 0; feature < feature_count; ++feature) {
            result(sample, feature) = X[sample * feature_count + feature];
        }
    }
    return result;
}

int ml_train(
    MLModel* model,
    const double* X,
    const int* y,
    int sample_count
) {
    g_last_error.clear();
    if (model == nullptr || X == nullptr || y == nullptr || sample_count <= 0) {
        set_error("Arguments d'entrainement invalides");
        return 0;
    }
    for (int sample = 0; sample < sample_count; ++sample) {
        if (y[sample] < 0 || y[sample] >= model->class_count) {
            set_error("Label hors limites");
            return 0;
        }
    }

    try {
        const Eigen::MatrixXd features =
            copy_features(X, sample_count, model->feature_count);
        for (int class_index = 0; class_index < model->class_count; ++class_index) {
            Eigen::VectorXi binary_labels(sample_count);
            const bool signed_labels =
                model->type == ML_PERCEPTRON || model->type == ML_SVM;
            for (int sample = 0; sample < sample_count; ++sample) {
                binary_labels(sample) = y[sample] == class_index
                    ? 1
                    : (signed_labels ? -1 : 0);
            }
            switch (model->type) {
                case ML_PERCEPTRON:
                    model->perceptrons[class_index]->fit(features, binary_labels);
                    break;
                case ML_MLP:
                    model->mlps[class_index]->fit(features, binary_labels);
                    break;
                case ML_RBF:
                    model->rbfs[class_index]->fit(features, binary_labels);
                    break;
                case ML_SVM:
                    model->svms[class_index]->fit(features, binary_labels);
                    break;
            }
        }
        return 1;
    } catch (const std::exception& exception) {
        set_error(exception.what());
        return 0;
    }
}

static double class_score(
    const MLModel& model,
    int class_index,
    const Eigen::VectorXd& x
) {
    switch (model.type) {
        case ML_PERCEPTRON:
            return model.perceptrons[class_index]->decisionFunction(x);
        case ML_MLP:
            return model.mlps[class_index]->predictProba(x);
        case ML_RBF:
            return model.rbfs[class_index]->predictProba(x);
        case ML_SVM:
            return model.svms[class_index]->decisionFunction(x);
    }
    return 0.0;
}

int ml_predict_with_score(
    const MLModel* model,
    const double* x,
    double* score
) {
    if (model == nullptr || x == nullptr) {
        set_error("Modele ou caracteristiques invalides");
        return -1;
    }
    Eigen::VectorXd features(model->feature_count);
    for (int feature = 0; feature < model->feature_count; ++feature) {
        features(feature) = x[feature];
    }
    int best_class = 0;
    double best_score = class_score(*model, 0, features);
    for (int class_index = 1; class_index < model->class_count; ++class_index) {
        const double score = class_score(*model, class_index, features);
        if (score > best_score) {
            best_score = score;
            best_class = class_index;
        }
    }
    if (score != nullptr) {
        *score = best_score;
    }
    return best_class;
}

int ml_predict(const MLModel* model, const double* x) {
    return ml_predict_with_score(model, x, nullptr);
}

double ml_score(
    const MLModel* model,
    const double* X,
    const int* y,
    int sample_count
) {
    if (model == nullptr || X == nullptr || y == nullptr || sample_count <= 0) {
        return 0.0;
    }
    int correct = 0;
    for (int sample = 0; sample < sample_count; ++sample) {
        if (ml_predict(model, X + sample * model->feature_count) == y[sample]) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / sample_count;
}

static void save_vector(std::ostream& file, const Eigen::VectorXd& values) {
    file << values.size();
    for (Eigen::Index i = 0; i < values.size(); ++i) file << ' ' << values(i);
    file << '\n';
}

static void save_matrix(std::ostream& file, const Eigen::MatrixXd& values) {
    file << values.rows() << ' ' << values.cols();
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        for (Eigen::Index col = 0; col < values.cols(); ++col) {
            file << ' ' << values(row, col);
        }
    }
    file << '\n';
}

static bool load_vector(std::istream& file, Eigen::VectorXd& values) {
    Eigen::Index size;
    if (!(file >> size) || size < 0) return false;
    values.resize(size);
    for (Eigen::Index i = 0; i < size; ++i) if (!(file >> values(i))) return false;
    return true;
}

static bool load_matrix(std::istream& file, Eigen::MatrixXd& values) {
    Eigen::Index rows, cols;
    if (!(file >> rows >> cols) || rows < 0 || cols < 0) return false;
    values.resize(rows, cols);
    for (Eigen::Index row = 0; row < rows; ++row) {
        for (Eigen::Index col = 0; col < cols; ++col) {
            if (!(file >> values(row, col))) return false;
        }
    }
    return true;
}

int ml_save(const MLModel* model, const char* path) {
    g_last_error.clear();
    if (model == nullptr || path == nullptr) {
        set_error("Modele ou chemin invalide");
        return 0;
    }
    std::ofstream file(path);
    if (!file) {
        set_error("Impossible d'ouvrir le fichier modele");
        return 0;
    }
    file << std::setprecision(17);
    file << "MLMODEL 2\n";
    file << static_cast<int>(model->type) << ' ' << model->feature_count
         << ' ' << model->class_count << '\n';
    file << model->params.epochs << ' ' << model->params.learning_rate << ' '
         << model->params.hidden_size << ' ' << model->params.rbf_centers << ' '
         << model->params.rbf_sigma << ' ' << model->params.svm_lambda << ' '
         << model->params.seed << '\n';

    for (int class_index = 0; class_index < model->class_count; ++class_index) {
        switch (model->type) {
            case ML_PERCEPTRON:
                save_vector(file, model->perceptrons[class_index]->getWeights());
                file << model->perceptrons[class_index]->getBias() << '\n';
                break;
            case ML_MLP:
                save_matrix(file, model->mlps[class_index]->getInputWeights());
                save_vector(file, model->mlps[class_index]->getHiddenBias());
                save_vector(file, model->mlps[class_index]->getOutputWeights());
                file << model->mlps[class_index]->getOutputBias() << '\n';
                break;
            case ML_RBF:
                save_matrix(file, model->rbfs[class_index]->getCenters());
                save_vector(file, model->rbfs[class_index]->getWeights());
                file << model->rbfs[class_index]->getBias() << '\n';
                break;
            case ML_SVM:
                save_vector(file, model->svms[class_index]->getWeights());
                file << model->svms[class_index]->getBias() << '\n';
                break;
        }
    }
    if (!file) {
        set_error("Erreur pendant l'ecriture du modele");
        return 0;
    }
    return 1;
}

static bool load_version_2(std::istream& file, MLModel& model) {
    for (int class_index = 0; class_index < model.class_count; ++class_index) {
        double bias;
        if (model.type == ML_PERCEPTRON || model.type == ML_SVM) {
            Eigen::VectorXd weights;
            if (!load_vector(file, weights) || !(file >> bias)) return false;
            if (weights.size() != model.feature_count) return false;
            if (model.type == ML_PERCEPTRON) {
                model.perceptrons[class_index]->setParameters(weights, bias);
            } else {
                model.svms[class_index]->setParameters(weights, bias);
            }
        } else if (model.type == ML_MLP) {
            Eigen::MatrixXd W1;
            Eigen::VectorXd b1, W2;
            if (!load_matrix(file, W1) || !load_vector(file, b1)
                || !load_vector(file, W2) || !(file >> bias)) return false;
            if (W1.rows() != model.feature_count
                || W1.cols() != model.params.hidden_size
                || b1.size() != model.params.hidden_size
                || W2.size() != model.params.hidden_size) return false;
            model.mlps[class_index]->setParameters(W1, b1, W2, bias);
        } else {
            Eigen::MatrixXd centers;
            Eigen::VectorXd weights;
            if (!load_matrix(file, centers) || !load_vector(file, weights)
                || !(file >> bias)) return false;
            if (centers.cols() != model.feature_count
                || centers.rows() <= 0
                || weights.size() != centers.rows()) return false;
            model.rbfs[class_index]->setParameters(centers, weights, bias);
        }
    }
    return true;
}

static bool load_legacy_vector(std::istream& file, std::vector<double>& values) {
    std::size_t size;
    if (!(file >> size)) return false;
    values.resize(size);
    for (double& value : values) if (!(file >> value)) return false;
    return true;
}

static bool load_version_1(std::istream& file, MLModel& model) {
    std::vector<double> weights, biases, W1raw, b1raw, W2raw, b2raw, centersraw;
    if (!load_legacy_vector(file, weights) || !load_legacy_vector(file, biases)
        || !load_legacy_vector(file, W1raw) || !load_legacy_vector(file, b1raw)
        || !load_legacy_vector(file, W2raw) || !load_legacy_vector(file, b2raw)
        || !load_legacy_vector(file, centersraw)) return false;

    if (model.type == ML_PERCEPTRON || model.type == ML_SVM) {
        if (weights.size() != static_cast<std::size_t>(model.feature_count * model.class_count)
            || biases.size() != static_cast<std::size_t>(model.class_count)) return false;
        for (int c = 0; c < model.class_count; ++c) {
            Eigen::VectorXd w(model.feature_count);
            for (int f = 0; f < model.feature_count; ++f) {
                w(f) = weights[c * model.feature_count + f];
            }
            if (model.type == ML_PERCEPTRON) model.perceptrons[c]->setParameters(w, biases[c]);
            else model.svms[c]->setParameters(w, biases[c]);
        }
    } else if (model.type == ML_MLP) {
        if (W1raw.size() != static_cast<std::size_t>(model.feature_count * model.params.hidden_size)
            || b1raw.size() != static_cast<std::size_t>(model.params.hidden_size)
            || W2raw.size() != static_cast<std::size_t>(model.params.hidden_size * model.class_count)
            || b2raw.size() != static_cast<std::size_t>(model.class_count)) return false;
        Eigen::MatrixXd W1(model.feature_count, model.params.hidden_size);
        Eigen::VectorXd b1(model.params.hidden_size);
        for (int f = 0; f < model.feature_count; ++f)
            for (int h = 0; h < model.params.hidden_size; ++h)
                W1(f, h) = W1raw[f * model.params.hidden_size + h];
        for (int h = 0; h < model.params.hidden_size; ++h) b1(h) = b1raw[h];
        for (int c = 0; c < model.class_count; ++c) {
            Eigen::VectorXd W2(model.params.hidden_size);
            for (int h = 0; h < model.params.hidden_size; ++h)
                W2(h) = W2raw[h * model.class_count + c];
            model.mlps[c]->setParameters(W1, b1, W2, b2raw[c]);
        }
    } else {
        const int center_count = static_cast<int>(centersraw.size()) / model.feature_count;
        if (center_count != model.params.rbf_centers
            || weights.size() != static_cast<std::size_t>(model.class_count * model.params.rbf_centers)
            || biases.size() != static_cast<std::size_t>(model.class_count)) return false;
        Eigen::MatrixXd centers(center_count, model.feature_count);
        for (int r = 0; r < center_count; ++r)
            for (int f = 0; f < model.feature_count; ++f)
                centers(r, f) = centersraw[r * model.feature_count + f];
        for (int c = 0; c < model.class_count; ++c) {
            Eigen::VectorXd w(center_count);
            for (int r = 0; r < center_count; ++r)
                w(r) = weights[c * model.params.rbf_centers + r];
            model.rbfs[c]->setParameters(centers, w, biases[c]);
        }
    }
    return true;
}

MLModel* ml_load(const char* path) {
    g_last_error.clear();
    if (path == nullptr) {
        set_error("Chemin de modele invalide");
        return nullptr;
    }
    std::ifstream file(path);
    std::string magic;
    int version, raw_type, feature_count, class_count;
    MLParams params = {};
    if (!(file >> magic >> version) || magic != "MLMODEL"
        || (version != 1 && version != 2)) {
        set_error("Format de modele invalide");
        return nullptr;
    }
    if (!(file >> raw_type >> feature_count >> class_count
          >> params.epochs >> params.learning_rate >> params.hidden_size
          >> params.rbf_centers >> params.rbf_sigma >> params.svm_lambda
          >> params.seed)) {
        set_error("En-tete du modele incomplet");
        return nullptr;
    }
    std::unique_ptr<MLModel> model(ml_create(
        static_cast<MLModelType>(raw_type), feature_count, class_count, params));
    if (!model) return nullptr;
    const bool complete = version == 2
        ? load_version_2(file, *model)
        : load_version_1(file, *model);
    if (!complete) {
        set_error("Fichier de modele incomplet ou dimensions incoherentes");
        return nullptr;
    }
    return model.release();
}

MLModelType ml_type(const MLModel* model) {
    return model == nullptr ? static_cast<MLModelType>(0) : model->type;
}

int ml_feature_count(const MLModel* model) {
    return model == nullptr ? 0 : model->feature_count;
}

int ml_class_count(const MLModel* model) {
    return model == nullptr ? 0 : model->class_count;
}

const char* ml_last_error(void) { return g_last_error.c_str(); }

void ml_free(MLModel* model) { delete model; }
