#include "dataset.h"
#include "ml_api.h"
#include "tests.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Dataset dataset = {0};
static Dataset train_dataset = {0};
static Dataset test_dataset = {0};
static MLModel* current_model = NULL;

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
            return "Aucun";
    }
}

static void clear_split(void) {
    dataset_free(&train_dataset);
    dataset_free(&test_dataset);
}

static void clear_everything(void) {
    ml_free(current_model);
    current_model = NULL;
    dataset_free(&dataset);
    clear_split();
}

static int load_dataset_from_path(const char* path) {
    dataset_free(&dataset);
    clear_split();

    if (!dataset_load_csv(path, &dataset)) {
        printf("Impossible de charger le CSV : %s\n", path);
        return 0;
    }

    if (!dataset_split(
            &dataset,
            0.20,
            42,
            &train_dataset,
            &test_dataset)) {
        printf("Impossible de separer le dataset.\n");
        dataset_free(&dataset);
        return 0;
    }

    dataset_print_info(&dataset);
    printf("Train : %d exemples\n", train_dataset.sample_count);
    printf("Test  : %d exemples\n", test_dataset.sample_count);
    return 1;
}

static void ask_and_load_dataset(void) {
    char path[512];

    printf("Chemin du CSV : ");
    scanf(" %511s", path);
    load_dataset_from_path(path);
}

static MLParams ask_model_params(MLModelType type) {
    MLParams params = ml_default_params();

    printf("Nombre d'epoques : ");
    scanf("%d", &params.epochs);

    printf("Learning rate : ");
    scanf("%lf", &params.learning_rate);

    if (type == ML_MLP) {
        printf("Nombre de neurones caches : ");
        scanf("%d", &params.hidden_size);
    }

    if (type == ML_RBF) {
        printf("Nombre de centres : ");
        scanf("%d", &params.rbf_centers);

        printf("Sigma : ");
        scanf("%lf", &params.rbf_sigma);
    }

    if (type == ML_SVM) {
        printf("Lambda de regularisation : ");
        scanf("%lf", &params.svm_lambda);
    }

    return params;
}

static int train_current_model(
    MLModelType type,
    MLParams params,
    int print_result
) {
    if (train_dataset.X == NULL) {
        printf("Chargez d'abord un dataset.\n");
        return 0;
    }

    MLModel* new_model = ml_create(
        type,
        train_dataset.feature_count,
        train_dataset.class_count,
        params
    );

    if (new_model == NULL) {
        printf("Creation impossible : %s\n", ml_last_error());
        return 0;
    }

    if (!ml_train(
            new_model,
            train_dataset.X,
            train_dataset.y,
            train_dataset.sample_count)) {
        printf("Entrainement impossible : %s\n", ml_last_error());
        ml_free(new_model);
        return 0;
    }

    ml_free(current_model);
    current_model = new_model;

    if (print_result) {
        printf("Modele %s entraine.\n", model_name(type));
        printf(
            "Accuracy train : %.3f\n",
            ml_score(
                current_model,
                train_dataset.X,
                train_dataset.y,
                train_dataset.sample_count
            )
        );
        printf(
            "Accuracy test  : %.3f\n",
            ml_score(
                current_model,
                test_dataset.X,
                test_dataset.y,
                test_dataset.sample_count
            )
        );
    }

    return 1;
}

static void train_from_menu(MLModelType type) {
    MLParams params = ask_model_params(type);
    train_current_model(type, params, 1);
}

static MLParams comparison_params(MLModelType type) {
    MLParams params = ml_default_params();

    if (type == ML_PERCEPTRON) {
        params.epochs = 30;
        params.learning_rate = 0.01;
    } else if (type == ML_MLP) {
        params.epochs = 300;
        params.learning_rate = 0.05;
        params.hidden_size = 32;
    } else if (type == ML_RBF) {
        params.epochs = 200;
        params.learning_rate = 0.05;
        params.rbf_centers = 20;
        params.rbf_sigma = 1.0;
    } else if (type == ML_SVM) {
        params.epochs = 100;
        params.learning_rate = 0.01;
        params.svm_lambda = 0.001;
    }

    return params;
}

static void compare_all_models(void) {
    if (train_dataset.X == NULL) {
        printf("Chargez d'abord un dataset.\n");
        return;
    }

    printf("\n%-12s | %-10s | %-10s\n", "Modele", "Train", "Test");
    printf("-------------+------------+-----------\n");

    for (int raw_type = ML_PERCEPTRON; raw_type <= ML_SVM; raw_type++) {
        MLModelType type = (MLModelType)raw_type;
        MLParams params = comparison_params(type);
        MLModel* model = ml_create(
            type,
            train_dataset.feature_count,
            train_dataset.class_count,
            params
        );

        if (model == NULL
            || !ml_train(
                model,
                train_dataset.X,
                train_dataset.y,
                train_dataset.sample_count)) {
            printf("%-12s | ERREUR\n", model_name(type));
            ml_free(model);
            continue;
        }

        double train_score = ml_score(
            model,
            train_dataset.X,
            train_dataset.y,
            train_dataset.sample_count
        );
        double test_score = ml_score(
            model,
            test_dataset.X,
            test_dataset.y,
            test_dataset.sample_count
        );

        printf(
            "%-12s | %-10.3f | %-10.3f\n",
            model_name(type),
            train_score,
            test_score
        );

        ml_free(model);
    }
}

static void evaluate_current_model(void) {
    if (current_model == NULL || test_dataset.X == NULL) {
        printf("Il faut un modele et un dataset charges.\n");
        return;
    }

    printf("Modele : %s\n", model_name(ml_type(current_model)));
    printf(
        "Accuracy train : %.3f\n",
        ml_score(
            current_model,
            train_dataset.X,
            train_dataset.y,
            train_dataset.sample_count
        )
    );
    printf(
        "Accuracy test  : %.3f\n",
        ml_score(
            current_model,
            test_dataset.X,
            test_dataset.y,
            test_dataset.sample_count
        )
    );
}

static void save_current_model(void) {
    char path[512];

    if (current_model == NULL) {
        printf("Aucun modele a sauvegarder.\n");
        return;
    }

    printf("Fichier de destination : ");
    scanf(" %511s", path);

    if (ml_save(current_model, path)) {
        printf("Modele sauvegarde.\n");
    } else {
        printf("Erreur : %s\n", ml_last_error());
    }
}

static void load_saved_model(void) {
    char path[512];
    MLModel* loaded;

    printf("Fichier modele : ");
    scanf(" %511s", path);
    loaded = ml_load(path);

    if (loaded == NULL) {
        printf("Erreur : %s\n", ml_last_error());
        return;
    }

    ml_free(current_model);
    current_model = loaded;

    printf("Modele charge : %s\n", model_name(ml_type(current_model)));
    printf("Caracteristiques : %d\n", ml_feature_count(current_model));
    printf("Classes : %d\n", ml_class_count(current_model));
}

static void predict_manually(void) {
    if (current_model == NULL) {
        printf("Chargez ou entrainez d'abord un modele.\n");
        return;
    }

    int feature_count = ml_feature_count(current_model);
    double* values = malloc(sizeof(double) * feature_count);

    if (values == NULL) {
        printf("Allocation impossible.\n");
        return;
    }

    for (int feature = 0; feature < feature_count; feature++) {
        printf("x[%d] : ", feature);
        scanf("%lf", &values[feature]);
    }

    printf("Classe predite : %d\n", ml_predict(current_model, values));
    free(values);
}

static void print_menu(void) {
    printf("\n=== DEMONSTRATION ML C / C++ ===\n");
    printf("1. Lancer les cas de test\n");
    printf("2. Charger un dataset CSV\n");
    printf("3. Afficher les informations du dataset\n");
    printf("4. Entrainer le perceptron\n");
    printf("5. Entrainer le PMC\n");
    printf("6. Entrainer le RBF\n");
    printf("7. Entrainer le SVM\n");
    printf("8. Comparer les quatre modeles\n");
    printf("9. Evaluer le modele courant\n");
    printf("10. Sauvegarder le modele courant\n");
    printf("11. Charger un modele sauvegarde\n");
    printf("12. Predire une nouvelle donnee\n");
    printf("0. Quitter\n");
    printf("Choix : ");
}

static int run_compare_command(const char* csv_path) {
    if (!load_dataset_from_path(csv_path)) {
        return 1;
    }

    compare_all_models();
    return 0;
}

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "--tests") == 0) {
        run_all_tests();
        return 0;
    }

    if (argc == 3 && strcmp(argv[1], "--compare") == 0) {
        int result = run_compare_command(argv[2]);
        clear_everything();
        return result;
    }

    int choice = -1;

    while (choice != 0) {
        print_menu();

        if (scanf("%d", &choice) != 1) {
            break;
        }

        switch (choice) {
            case 1:
                run_all_tests();
                break;
            case 2:
                ask_and_load_dataset();
                break;
            case 3:
                if (dataset.X == NULL) {
                    printf("Aucun dataset charge.\n");
                } else {
                    dataset_print_info(&dataset);
                }
                break;
            case 4:
                train_from_menu(ML_PERCEPTRON);
                break;
            case 5:
                train_from_menu(ML_MLP);
                break;
            case 6:
                train_from_menu(ML_RBF);
                break;
            case 7:
                train_from_menu(ML_SVM);
                break;
            case 8:
                compare_all_models();
                break;
            case 9:
                evaluate_current_model();
                break;
            case 10:
                save_current_model();
                break;
            case 11:
                load_saved_model();
                break;
            case 12:
                predict_manually();
                break;
            case 0:
                break;
            default:
                printf("Choix inconnu.\n");
        }
    }

    clear_everything();
    return 0;
}
