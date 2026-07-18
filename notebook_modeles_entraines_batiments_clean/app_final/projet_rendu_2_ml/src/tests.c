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
    printf("  rechargement : OK, accuracy test %.1f %%\n",
           score_after_load * 100.0);

    ml_free(loaded);
    remove(path);
}

static void run_case(
    const char* title,
    const double* train_X,
    const int* train_y,
    int train_count,
    const double* test_X,
    const int* test_y,
    int test_count,
    int feature_count,
    int class_count
) {
    /* Train et test separes : points test jamais appris. */
    printf("\n=== %s ===\n", title);

    for (int raw_type = ML_PERCEPTRON; raw_type <= ML_SVM; raw_type++) {
        MLModelType type = (MLModelType)raw_type;
        MLParams params = test_params(type, train_count);
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

        if (!ml_train(model, train_X, train_y, train_count)) {
            printf("%-12s entrainement impossible : %s\n",
                   model_name(type),
                   ml_last_error());
            ml_free(model);
            continue;
        }

        printf("%-12s train : %5.1f %% | test jamais vu : %5.1f %%\n",
               model_name(type),
               ml_score(model, train_X, train_y, train_count) * 100.0,
               ml_score(model, test_X, test_y, test_count) * 100.0);

        test_save_and_load(type, model, test_X, test_y, test_count);
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

    const double linear_test_X[] = {
        1.2, 1.4,
        1.8, 1.6,
        2.1, 2.2,
        5.2, 5.4,
        5.8, 5.3,
        5.7, 6.2
    };
    const int linear_test_y[] = {0, 0, 0, 1, 1, 1};

    const double xor_X[] = {
        0.00, 0.00,
        0.05, 0.10,
        0.12, 0.04,
        0.00, 1.00,
        0.08, 0.90,
        0.12, 0.96,
        1.00, 0.00,
        0.90, 0.08,
        0.96, 0.14,
        1.00, 1.00,
        0.90, 0.92,
        0.96, 0.86
    };
    const int xor_y[] = {0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0};

    const double xor_test_X[] = {
        0.03, 0.06,
        0.15, 0.12,
        0.04, 0.94,
        0.15, 0.88,
        0.94, 0.04,
        0.86, 0.16,
        0.94, 0.96,
        0.86, 0.88
    };
    const int xor_test_y[] = {0, 0, 1, 1, 1, 1, 0, 0};

    const double three_class_X[] = {
        0.0, 0.0,
        0.2, 0.1,
        4.0, 0.0,
        4.2, 0.1,
        2.0, 4.0,
        2.1, 4.2
    };
    const int three_class_y[] = {0, 0, 1, 1, 2, 2};

    const double three_class_test_X[] = {
        0.1, 0.2,
        0.3, 0.0,
        3.9, 0.2,
        4.3, 0.0,
        1.9, 3.8,
        2.2, 4.1
    };
    const int three_class_test_y[] = {0, 0, 1, 1, 2, 2};

    run_case(
        "Donnees lineairement separables",
        linear_X,
        linear_y,
        8,
        linear_test_X,
        linear_test_y,
        6,
        2,
        2
    );

    run_case(
        "XOR non lineaire",
        xor_X,
        xor_y,
        12,
        xor_test_X,
        xor_test_y,
        8,
        2,
        2
    );

    run_case(
        "Classification one-vs-rest a trois classes",
        three_class_X,
        three_class_y,
        6,
        three_class_test_X,
        three_class_test_y,
        6,
        2,
        3
    );
}
