#include <stdio.h>
#include "../include/rbf.h"

void init_rbf(RBFNetwork *model, int input_size, int num_centers, double sigma) {
    model->input_size = input_size;
    model->num_centers = num_centers;
    model->sigma = sigma;
}

void fit_rbf(RBFNetwork *model) {
    printf("fit_rbf non encore implemente\n");
}

void predict_rbf(RBFNetwork *model) {
    printf("predict_rbf non encore implemente\n");
}

double score_rbf(RBFNetwork *model) {
    printf("score_rbf non encore implemente\n");
    return 0.0;
}