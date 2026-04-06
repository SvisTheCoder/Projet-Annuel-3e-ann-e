#include <stdio.h>
#include <stdlib.h>
#include "../include/perceptron.h"

void init_perceptron(RosenblattPerceptron *model, double learning_rate, int epochs) {
    model->learning_rate = learning_rate;
    model->epochs = epochs;
    model->weights[0] = 0.0;
    model->weights[1] = 0.0;
    model->bias = 0.0;

    model->errors_per_epoch = (int *)malloc(sizeof(int) * epochs);
    if (model->errors_per_epoch != NULL) {
        for (int i = 0; i < epochs; i++) {
            model->errors_per_epoch[i] = 0;
        }
    }
}

void free_perceptron(RosenblattPerceptron *model) {
    if (model->errors_per_epoch != NULL) {
        free(model->errors_per_epoch);
        model->errors_per_epoch = NULL;
    }
}

int predict_single_perceptron(RosenblattPerceptron *model, double x[2]) {
    double result = x[0] * model->weights[0]
                  + x[1] * model->weights[1]
                  + model->bias;

    if (result >= 0) {
        return 1;
    } else {
        return -1;
    }
}

void fit_perceptron(RosenblattPerceptron *model, double X[][2], int y[], int n_samples) {
    for (int epoch = 0; epoch < model->epochs; epoch++) {
        int errors = 0;

        for (int i = 0; i < n_samples; i++) {
            int prediction = predict_single_perceptron(model, X[i]);

            if (prediction != y[i]) {
                model->weights[0] = model->weights[0] + model->learning_rate * y[i] * X[i][0];
                model->weights[1] = model->weights[1] + model->learning_rate * y[i] * X[i][1];
                model->bias = model->bias + model->learning_rate * y[i];
                errors++;
            }
        }

        if (model->errors_per_epoch != NULL) {
            model->errors_per_epoch[epoch] = errors;
        }
    }
}

double score_perceptron(RosenblattPerceptron *model, double X[][2], int y[], int n_samples) {
    int correct = 0;

    for (int i = 0; i < n_samples; i++) {
        int prediction = predict_single_perceptron(model, X[i]);
        if (prediction == y[i]) {
            correct++;
        }
    }

    return (double)correct / n_samples;
}

int save_perceptron(RosenblattPerceptron *model, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        return 0;
    }

    fprintf(file, "%lf\n", model->learning_rate);
    fprintf(file, "%d\n", model->epochs);
    fprintf(file, "%lf %lf\n", model->weights[0], model->weights[1]);
    fprintf(file, "%lf\n", model->bias);

    fclose(file);
    return 1;
}

int load_perceptron(RosenblattPerceptron *model, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        return 0;
    }

    fscanf(file, "%lf", &model->learning_rate);
    fscanf(file, "%d", &model->epochs);
    fscanf(file, "%lf %lf", &model->weights[0], &model->weights[1]);
    fscanf(file, "%lf", &model->bias);

    model->errors_per_epoch = NULL;

    fclose(file);
    return 1;
}