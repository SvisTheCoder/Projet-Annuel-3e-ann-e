#ifndef MLP_H
#define MLP_H

typedef struct {
    int input_size;
    int hidden_size;
    double learning_rate;
    int epochs;
} MLPClassifier;

void init_mlp(MLPClassifier *model, int input_size, int hidden_size, double learning_rate, int epochs);
void fit_mlp(MLPClassifier *model);
void predict_mlp(MLPClassifier *model);
double score_mlp(MLPClassifier *model);

#endif