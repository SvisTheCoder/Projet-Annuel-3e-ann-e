#include "ml_api.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

/*
 * Cette partie est écrite en C++ mais n'expose que les fonctions C de ml_api.h.
 * L'application C manipule donc un simple pointeur MLModel* sans voir les
 * std::vector ni les détails des modèles.
 */

static std::string g_last_error;

static void set_error(const std::string& message) {
    g_last_error = message;
}

static double dot_product(const double* a, const double* b, int size) {
    double result = 0.0;

    for (int i = 0; i < size; i++) {
        result += a[i] * b[i];
    }

    return result;
}

static double sigmoid(double value) {
    if (value > 50.0) {
        return 1.0;
    }

    if (value < -50.0) {
        return 0.0;
    }

    return 1.0 / (1.0 + std::exp(-value));
}

static void softmax(std::vector<double>& values) {
    double maximum = *std::max_element(values.begin(), values.end());
    double total = 0.0;

    for (double& value : values) {
        value = std::exp(value - maximum);
        total += value;
    }

    for (double& value : values) {
        value /= total;
    }
}

static int index_of_maximum(const std::vector<double>& values) {
    return static_cast<int>(
        std::max_element(values.begin(), values.end()) - values.begin()
    );
}

struct MLModel {
    MLModelType type;
    int feature_count;
    int class_count;
    MLParams params;

    /* Perceptron et SVM : une ligne de poids par classe. */
    std::vector<double> weights;
    std::vector<double> biases;

    /* PMC : entrée -> couche cachée -> sortie. */
    std::vector<double> weights_input_hidden;
    std::vector<double> biases_hidden;
    std::vector<double> weights_hidden_output;
    std::vector<double> biases_output;

    /* RBF : centres utilisés par les activations gaussiennes. */
    std::vector<double> centers;
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

MLModel* ml_create(
    MLModelType type,
    int feature_count,
    int class_count,
    MLParams params
) {
    if (feature_count <= 0 || class_count < 2) {
        set_error("Dimensions invalides");
        return nullptr;
    }

    if (params.epochs <= 0 || params.learning_rate <= 0.0) {
        set_error("Parametres d'apprentissage invalides");
        return nullptr;
    }

    try {
        MLModel* model = new MLModel();
        model->type = type;
        model->feature_count = feature_count;
        model->class_count = class_count;
        model->params = params;

        if (type == ML_PERCEPTRON || type == ML_SVM) {
            model->weights.assign(class_count * feature_count, 0.0);
            model->biases.assign(class_count, 0.0);
        } else if (type == ML_MLP) {
            if (model->params.hidden_size <= 0) {
                model->params.hidden_size = 1;
            }

            int hidden_size = model->params.hidden_size;
            std::mt19937 generator(model->params.seed);
            std::normal_distribution<double> random_weight(0.0, 0.1);

            model->weights_input_hidden.resize(feature_count * hidden_size);
            model->biases_hidden.assign(hidden_size, 0.0);
            model->weights_hidden_output.resize(hidden_size * class_count);
            model->biases_output.assign(class_count, 0.0);

            for (double& weight : model->weights_input_hidden) {
                weight = random_weight(generator);
            }

            for (double& weight : model->weights_hidden_output) {
                weight = random_weight(generator);
            }
        } else if (type == ML_RBF) {
            if (model->params.rbf_centers <= 0) {
                model->params.rbf_centers = 1;
            }

            if (model->params.rbf_sigma <= 0.0) {
                delete model;
                set_error("Sigma RBF doit etre positif");
                return nullptr;
            }

            model->weights.assign(
                class_count * model->params.rbf_centers,
                0.0
            );
            model->biases.assign(class_count, 0.0);
        } else {
            delete model;
            set_error("Type de modele inconnu");
            return nullptr;
        }

        return model;
    } catch (...) {
        set_error("Allocation memoire impossible");
        return nullptr;
    }
}

static std::vector<double> linear_scores(
    const MLModel* model,
    const double* x
) {
    std::vector<double> scores(model->class_count, 0.0);

    for (int class_index = 0;
         class_index < model->class_count;
         class_index++) {
        const double* class_weights =
            &model->weights[class_index * model->feature_count];

        scores[class_index] =
            dot_product(x, class_weights, model->feature_count)
            + model->biases[class_index];
    }

    return scores;
}

static std::vector<double> mlp_scores(
    const MLModel* model,
    const double* x
) {
    int hidden_size = model->params.hidden_size;
    std::vector<double> hidden(hidden_size, 0.0);
    std::vector<double> scores(model->class_count, 0.0);

    for (int hidden_index = 0;
         hidden_index < hidden_size;
         hidden_index++) {
        double value = model->biases_hidden[hidden_index];

        for (int feature = 0;
             feature < model->feature_count;
             feature++) {
            value += x[feature]
                * model->weights_input_hidden[
                    feature * hidden_size + hidden_index
                ];
        }

        hidden[hidden_index] = sigmoid(value);
    }

    for (int class_index = 0;
         class_index < model->class_count;
         class_index++) {
        double value = model->biases_output[class_index];

        for (int hidden_index = 0;
             hidden_index < hidden_size;
             hidden_index++) {
            value += hidden[hidden_index]
                * model->weights_hidden_output[
                    hidden_index * model->class_count + class_index
                ];
        }

        scores[class_index] = value;
    }

    return scores;
}

static double rbf_activation(
    const MLModel* model,
    const double* x,
    int center_index
) {
    double squared_distance = 0.0;
    const double* center =
        &model->centers[center_index * model->feature_count];

    for (int feature = 0;
         feature < model->feature_count;
         feature++) {
        double difference = x[feature] - center[feature];
        squared_distance += difference * difference;
    }

    double sigma = model->params.rbf_sigma;
    return std::exp(-squared_distance / (2.0 * sigma * sigma));
}

static std::vector<double> rbf_scores(
    const MLModel* model,
    const double* x
) {
    int available_centers = static_cast<int>(
        model->centers.size() / model->feature_count
    );
    std::vector<double> scores(model->class_count, 0.0);

    for (int class_index = 0;
         class_index < model->class_count;
         class_index++) {
        double value = model->biases[class_index];

        for (int center_index = 0;
             center_index < available_centers;
             center_index++) {
            double activation = rbf_activation(
                model,
                x,
                center_index
            );

            value += model->weights[
                class_index * model->params.rbf_centers + center_index
            ] * activation;
        }

        scores[class_index] = value;
    }

    return scores;
}

static std::vector<double> model_scores(
    const MLModel* model,
    const double* x
) {
    if (model->type == ML_PERCEPTRON || model->type == ML_SVM) {
        return linear_scores(model, x);
    }

    if (model->type == ML_MLP) {
        return mlp_scores(model, x);
    }

    return rbf_scores(model, x);
}

static int train_perceptron(
    MLModel* model,
    const double* X,
    const int* y,
    int sample_count
) {
    double learning_rate = model->params.learning_rate;

    /*
     * Strategie un-contre-tous : pour chaque classe, la vraie classe vaut +1
     * et toutes les autres valent -1. C'est la regle de Rosenblatt expliquee
     * dans le cours et dans le rapport.
     */
    for (int epoch = 0; epoch < model->params.epochs; epoch++) {
        for (int sample = 0; sample < sample_count; sample++) {
            const double* x = &X[sample * model->feature_count];

            for (int class_index = 0;
                 class_index < model->class_count;
                 class_index++) {
                double expected = (y[sample] == class_index) ? 1.0 : -1.0;
                double* class_weights =
                    &model->weights[class_index * model->feature_count];
                double score =
                    dot_product(x, class_weights, model->feature_count)
                    + model->biases[class_index];
                double predicted = (score >= 0.0) ? 1.0 : -1.0;

                if (predicted != expected) {
                    for (int feature = 0;
                         feature < model->feature_count;
                         feature++) {
                        class_weights[feature] +=
                            learning_rate * expected * x[feature];
                    }

                    model->biases[class_index] +=
                        learning_rate * expected;
                }
            }
        }
    }

    return 1;
}

static int train_svm(
    MLModel* model,
    const double* X,
    const int* y,
    int sample_count
) {
    double learning_rate = model->params.learning_rate;
    double lambda = model->params.svm_lambda;

    /* SVM lineaire un-contre-tous, perte hinge et regularisation L2. */
    for (int epoch = 0; epoch < model->params.epochs; epoch++) {
        for (int sample = 0; sample < sample_count; sample++) {
            const double* x = &X[sample * model->feature_count];

            for (int class_index = 0;
                 class_index < model->class_count;
                 class_index++) {
                double label = (y[sample] == class_index) ? 1.0 : -1.0;
                double* class_weights =
                    &model->weights[class_index * model->feature_count];
                double score =
                    dot_product(x, class_weights, model->feature_count)
                    + model->biases[class_index];
                double margin = label * score;

                for (int feature = 0;
                     feature < model->feature_count;
                     feature++) {
                    double gradient = 2.0 * lambda * class_weights[feature];

                    if (margin < 1.0) {
                        gradient -= label * x[feature];
                    }

                    class_weights[feature] -= learning_rate * gradient;
                }

                if (margin < 1.0) {
                    model->biases[class_index] += learning_rate * label;
                }
            }
        }
    }

    return 1;
}

static int train_mlp(
    MLModel* model,
    const double* X,
    const int* y,
    int sample_count
) {
    int hidden_size = model->params.hidden_size;
    double learning_rate = model->params.learning_rate;
    std::mt19937 generator(model->params.seed);
    std::vector<int> order(sample_count);

    for (int i = 0; i < sample_count; i++) {
        order[i] = i;
    }

    for (int epoch = 0; epoch < model->params.epochs; epoch++) {
        std::shuffle(order.begin(), order.end(), generator);

        for (int ordered_index : order) {
            const double* x =
                &X[ordered_index * model->feature_count];
            std::vector<double> hidden(hidden_size, 0.0);
            std::vector<double> output(model->class_count, 0.0);

            /* Propagation avant : entree vers couche cachee. */
            for (int hidden_index = 0;
                 hidden_index < hidden_size;
                 hidden_index++) {
                double value = model->biases_hidden[hidden_index];

                for (int feature = 0;
                     feature < model->feature_count;
                     feature++) {
                    value += x[feature]
                        * model->weights_input_hidden[
                            feature * hidden_size + hidden_index
                        ];
                }

                hidden[hidden_index] = sigmoid(value);
            }

            /* Couche cachee vers sorties, puis softmax. */
            for (int class_index = 0;
                 class_index < model->class_count;
                 class_index++) {
                double value = model->biases_output[class_index];

                for (int hidden_index = 0;
                     hidden_index < hidden_size;
                     hidden_index++) {
                    value += hidden[hidden_index]
                        * model->weights_hidden_output[
                            hidden_index * model->class_count + class_index
                        ];
                }

                output[class_index] = value;
            }

            softmax(output);

            /* Derivee de softmax + entropie croisee. */
            std::vector<double> output_error = output;
            output_error[y[ordered_index]] -= 1.0;
            std::vector<double> hidden_error(hidden_size, 0.0);

            /* Calculer l'erreur cachee avant de modifier W2. */
            for (int hidden_index = 0;
                 hidden_index < hidden_size;
                 hidden_index++) {
                for (int class_index = 0;
                     class_index < model->class_count;
                     class_index++) {
                    hidden_error[hidden_index] +=
                        output_error[class_index]
                        * model->weights_hidden_output[
                            hidden_index * model->class_count + class_index
                        ];
                }

                hidden_error[hidden_index] *=
                    hidden[hidden_index]
                    * (1.0 - hidden[hidden_index]);
            }

            /* Mise a jour SGD : un exemple a la fois. */
            for (int hidden_index = 0;
                 hidden_index < hidden_size;
                 hidden_index++) {
                for (int class_index = 0;
                     class_index < model->class_count;
                     class_index++) {
                    model->weights_hidden_output[
                        hidden_index * model->class_count + class_index
                    ] -= learning_rate
                        * hidden[hidden_index]
                        * output_error[class_index];
                }
            }

            for (int class_index = 0;
                 class_index < model->class_count;
                 class_index++) {
                model->biases_output[class_index] -=
                    learning_rate * output_error[class_index];
            }

            for (int feature = 0;
                 feature < model->feature_count;
                 feature++) {
                for (int hidden_index = 0;
                     hidden_index < hidden_size;
                     hidden_index++) {
                    model->weights_input_hidden[
                        feature * hidden_size + hidden_index
                    ] -= learning_rate
                        * x[feature]
                        * hidden_error[hidden_index];
                }
            }

            for (int hidden_index = 0;
                 hidden_index < hidden_size;
                 hidden_index++) {
                model->biases_hidden[hidden_index] -=
                    learning_rate * hidden_error[hidden_index];
            }
        }
    }

    return 1;
}

static int train_rbf(
    MLModel* model,
    const double* X,
    const int* y,
    int sample_count
) {
    int center_count = std::min(
        model->params.rbf_centers,
        sample_count
    );
    double learning_rate = model->params.learning_rate;

    /*
     * Version volontairement simple : on choisit des exemples regulierement
     * repartis dans le dataset comme centres au lieu d'utiliser K-means.
     */
    model->centers.assign(
        center_count * model->feature_count,
        0.0
    );

    for (int center_index = 0;
         center_index < center_count;
         center_index++) {
        int source_sample = center_index * sample_count / center_count;

        for (int feature = 0;
             feature < model->feature_count;
             feature++) {
            model->centers[
                center_index * model->feature_count + feature
            ] = X[source_sample * model->feature_count + feature];
        }
    }

    for (int epoch = 0; epoch < model->params.epochs; epoch++) {
        for (int sample = 0; sample < sample_count; sample++) {
            const double* x = &X[sample * model->feature_count];
            std::vector<double> activations(center_count, 0.0);
            std::vector<double> output(model->class_count, 0.0);

            for (int center_index = 0;
                 center_index < center_count;
                 center_index++) {
                activations[center_index] = rbf_activation(
                    model,
                    x,
                    center_index
                );
            }

            for (int class_index = 0;
                 class_index < model->class_count;
                 class_index++) {
                double value = model->biases[class_index];

                for (int center_index = 0;
                     center_index < center_count;
                     center_index++) {
                    value += model->weights[
                        class_index * model->params.rbf_centers
                        + center_index
                    ] * activations[center_index];
                }

                output[class_index] = value;
            }

            softmax(output);
            output[y[sample]] -= 1.0;

            for (int class_index = 0;
                 class_index < model->class_count;
                 class_index++) {
                for (int center_index = 0;
                     center_index < center_count;
                     center_index++) {
                    model->weights[
                        class_index * model->params.rbf_centers
                        + center_index
                    ] -= learning_rate
                        * output[class_index]
                        * activations[center_index];
                }

                model->biases[class_index] -=
                    learning_rate * output[class_index];
            }
        }
    }

    return 1;
}

int ml_train(
    MLModel* model,
    const double* X,
    const int* y,
    int sample_count
) {
    if (model == nullptr || X == nullptr || y == nullptr || sample_count <= 0) {
        set_error("Arguments d'entrainement invalides");
        return 0;
    }

    for (int sample = 0; sample < sample_count; sample++) {
        if (y[sample] < 0 || y[sample] >= model->class_count) {
            set_error("Label hors limites");
            return 0;
        }
    }

    if (model->type == ML_PERCEPTRON) {
        return train_perceptron(model, X, y, sample_count);
    }

    if (model->type == ML_MLP) {
        return train_mlp(model, X, y, sample_count);
    }

    if (model->type == ML_RBF) {
        return train_rbf(model, X, y, sample_count);
    }

    return train_svm(model, X, y, sample_count);
}

int ml_predict(const MLModel* model, const double* x) {
    if (model == nullptr || x == nullptr) {
        return -1;
    }

    return index_of_maximum(model_scores(model, x));
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

    for (int sample = 0; sample < sample_count; sample++) {
        const double* x = &X[sample * model->feature_count];

        if (ml_predict(model, x) == y[sample]) {
            correct++;
        }
    }

    return static_cast<double>(correct) / sample_count;
}

static void save_vector(
    std::ofstream& file,
    const std::vector<double>& values
) {
    file << values.size() << '\n';

    for (double value : values) {
        file << value << ' ';
    }

    file << '\n';
}

static bool load_vector(
    std::ifstream& file,
    std::vector<double>& values
) {
    std::size_t size = 0;

    if (!(file >> size)) {
        return false;
    }

    values.resize(size);

    for (double& value : values) {
        if (!(file >> value)) {
            return false;
        }
    }

    return true;
}

int ml_save(const MLModel* model, const char* path) {
    if (model == nullptr || path == nullptr) {
        set_error("Modele ou chemin invalide");
        return 0;
    }

    std::ofstream file(path);

    if (!file) {
        set_error("Impossible d'ouvrir le fichier modele");
        return 0;
    }

    file << "MLMODEL 1\n";
    file << static_cast<int>(model->type) << ' '
         << model->feature_count << ' '
         << model->class_count << '\n';
    file << model->params.epochs << ' '
         << model->params.learning_rate << ' '
         << model->params.hidden_size << ' '
         << model->params.rbf_centers << ' '
         << model->params.rbf_sigma << ' '
         << model->params.svm_lambda << ' '
         << model->params.seed << '\n';

    save_vector(file, model->weights);
    save_vector(file, model->biases);
    save_vector(file, model->weights_input_hidden);
    save_vector(file, model->biases_hidden);
    save_vector(file, model->weights_hidden_output);
    save_vector(file, model->biases_output);
    save_vector(file, model->centers);

    return 1;
}

MLModel* ml_load(const char* path) {
    if (path == nullptr) {
        set_error("Chemin de modele invalide");
        return nullptr;
    }

    std::ifstream file(path);
    std::string magic;
    int version = 0;
    int type = 0;
    int feature_count = 0;
    int class_count = 0;
    MLParams params = {};

    if (!(file >> magic >> version) || magic != "MLMODEL" || version != 1) {
        set_error("Format de modele invalide");
        return nullptr;
    }

    if (!(file >> type >> feature_count >> class_count
          >> params.epochs
          >> params.learning_rate
          >> params.hidden_size
          >> params.rbf_centers
          >> params.rbf_sigma
          >> params.svm_lambda
          >> params.seed)) {
        set_error("En-tete du modele incomplet");
        return nullptr;
    }

    MLModel* model = ml_create(
        static_cast<MLModelType>(type),
        feature_count,
        class_count,
        params
    );

    if (model == nullptr) {
        return nullptr;
    }

    bool complete =
        load_vector(file, model->weights)
        && load_vector(file, model->biases)
        && load_vector(file, model->weights_input_hidden)
        && load_vector(file, model->biases_hidden)
        && load_vector(file, model->weights_hidden_output)
        && load_vector(file, model->biases_output)
        && load_vector(file, model->centers);

    if (!complete) {
        delete model;
        set_error("Fichier de modele incomplet");
        return nullptr;
    }

    return model;
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

const char* ml_last_error(void) {
    return g_last_error.c_str();
}

void ml_free(MLModel* model) {
    delete model;
}
