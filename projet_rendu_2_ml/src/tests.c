#include "tests.h"
#include "ml_api.h"

#include <stdio.h>
#include <stdlib.h>

static const char* model_name(MLModelType type) {
    switch (type) {
        case ML_PERCEPTRON:
            return "Perceptron";
        case ML_MLP:
            return "PMC";
        case ML_RBF:
            return "RBF";
        case ML_SVM:
            return "SVM";
        default:
            return "Inconnu";
    }
}

static MLParams test_params(MLModelType type, int sample_count) {
    MLParams params = ml_default_params();

    params.hidden_size = 6;
    params.rbf_centers = sample_count;
    params.rbf_sigma = 0.8;

    if (type == ML_MLP || type == ML_RBF) {
        params.epochs = 1500;
        params.learning_rate = 0.1;
    } else {
        params.epochs = 100;
        params.learning_rate = 0.05;
    }

    return params;
}

static void test_save_and_load(
    MLModelType type,
    MLModel* model,
    const double* X,
    const int* y,
    int sample_count
) {
    char path[64];
    snprintf(path, sizeof(path), "test_model_%d.tmp", (int)type);

    if (!ml_save(model, path)) {
        printf("  sauvegarde : ECHEC (%s)\n", ml_last_error());
        return;
    }

    MLModel* loaded = ml_load(path);

    if (loaded == NULL) {
        printf("  rechargement : ECHEC (%s)\n", ml_last_error());
        remove(path);
        return;
    }

    double score_after_load = ml_score(loaded, X, y, sample_count);
    printf("  rechargement : OK, accuracy %.3f\n", score_after_load);

    ml_free(loaded);
    remove(path);
}

static void run_case(
    const char* title,
    const double* X,
    const int* y,
    int sample_count,
    int feature_count,
    int class_count
) {
    printf("\n=== %s ===\n", title);

    for (int raw_type = ML_PERCEPTRON; raw_type <= ML_SVM; raw_type++) {
        MLModelType type = (MLModelType)raw_type;
        MLParams params = test_params(type, sample_count);
        MLModel* model = ml_create(
            type,
            feature_count,
            class_count,
            params
        );

        if (model == NULL) {
            printf("%-12s creation impossible : %s\n",
                   model_name(type),
                   ml_last_error());
            continue;
        }

        if (!ml_train(model, X, y, sample_count)) {
            printf("%-12s entrainement impossible : %s\n",
                   model_name(type),
                   ml_last_error());
            ml_free(model);
            continue;
        }

        printf("%-12s accuracy : %.3f\n",
               model_name(type),
               ml_score(model, X, y, sample_count));

        test_save_and_load(type, model, X, y, sample_count);
        ml_free(model);
    }
}

void run_all_tests(void) {
    const double linear_X[] = {
        1.0, 1.0,
        1.5, 2.0,
        2.0, 1.0,
        2.0, 2.5,
        5.0, 5.0,
        6.0, 5.0,
        5.5, 6.0,
        6.0, 6.5
    };
    const int linear_y[] = {0, 0, 0, 0, 1, 1, 1, 1};

    const double xor_X[] = {
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0
    };
    const int xor_y[] = {0, 1, 1, 0};

    run_case(
        "Donnees lineairement separables",
        linear_X,
        linear_y,
        8,
        2,
        2
    );

    run_case(
        "XOR non lineaire",
        xor_X,
        xor_y,
        4,
        2,
        2
    );
}
