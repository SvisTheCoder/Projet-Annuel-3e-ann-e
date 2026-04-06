#ifndef RBF_H
#define RBF_H

typedef struct {
    int input_size;
    int num_centers;
    double sigma;
} RBFNetwork;

void init_rbf(RBFNetwork *model, int input_size, int num_centers, double sigma);
void fit_rbf(RBFNetwork *model);
void predict_rbf(RBFNetwork *model);
double score_rbf(RBFNetwork *model);

#endif