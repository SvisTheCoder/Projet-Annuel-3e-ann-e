#include <stdio.h>
#include "../include/svm.h"

void init_svm(SVMClassifier *model, double learning_rate, int epochs, double lambda) {
    model->learning_rate = learning_rate;
    model->epochs = epochs;
    model->lambda = lambda;
}
void fit_svm(SVMClassifier *model) {
    printf("fit_svm non encore implemente\n");
}
void predict_svm(SVMClassifier *model) {
    printf("predict_svm non encore implemente\n");
}
double score_svm(SVMClassifier *model) {
    printf("score_svm non encore implemente\n");
    return 0.0;
}