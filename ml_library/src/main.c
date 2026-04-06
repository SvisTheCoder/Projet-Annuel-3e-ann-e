#include <stdio.h>
#include "../include/perceptron.h"

int main() {
    double X[8][2] = {
        {1.0, 1.0},
        {1.5, 2.0},
        {2.0, 1.0},
        {2.0, 2.5},
        {5.0, 5.0},
        {6.0, 5.0},
        {5.5, 6.0},
        {6.0, 6.5}
    };

    int y[8] = {-1, -1, -1, -1, 1, 1, 1, 1};

    RosenblattPerceptron model;
    init_perceptron(&model, 0.1, 20);

    fit_perceptron(&model, X, y, 8);

    printf(" RESULTATS PERCEPTRON \n");
    printf("Poids : [%f, %f]\n", model.weights[0], model.weights[1]);
    printf("Biais : %f\n", model.bias);

    printf("Predictions :\n");
    for (int i = 0; i < 8; i++) {
        int prediction = predict_single_perceptron(&model, X[i]);
        printf("Point %d -> %d\n", i, prediction);
    }

    printf("Accuracy : %f\n", score_perceptron(&model, X, y, 8));

    printf("Erreurs par epoque :\n");
    for (int i = 0; i < model.epochs; i++) {
        printf("Epoch %d -> %d erreurs\n", i + 1, model.errors_per_epoch[i]);
    }

    if (save_perceptron(&model, "perceptron_model.txt")) {
        printf("Modele sauvegard dans perceptron_model.txt\n");
    } else {
        printf("Erreur lors de la sauvegarde\n");
    }
    free_perceptron(&model);
    return 0;
}