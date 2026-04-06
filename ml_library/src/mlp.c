#include <stdio.h>
#include "../include/mlp.h"

void init_mlp(MLPClassifier *model, int input_size, int hidden_size, double learning_rate, int epochs) {
    model->input_size = input_size;
    model->hidden_size = hidden_size;
    model->learning_rate = learning_rate;
    model->epochs = epochs;
}
void fit_mlp(MLPClassifier *model) {
    printf("fit_mlp non encore implemente\n");
}
void predict_mlp(MLPClassifier *model) {
    printf("predict_mlp non encore implemente\n");
}
double score_mlp(MLPClassifier *model) {
    printf("score_mlp non encore implemente\n");
    return 0.0;
}