#ifndef PERCEPTRON_H
#define PERCEPTRON_H

typedef struct {
    double learning_rate;
    int epochs;
    double weights[2];
    double bias;
    int *errors_per_epoch;
} RosenblattPerceptron;

void init_perceptron(RosenblattPerceptron *model, double learning_rate, int epochs);
void free_perceptron(RosenblattPerceptron *model);

int predict_single_perceptron(RosenblattPerceptron *model, double x[2]);
void fit_perceptron(RosenblattPerceptron *model, double X[][2], int y[], int n_samples);
double score_perceptron(RosenblattPerceptron *model, double X[][2], int y[], int n_samples);

int save_perceptron(RosenblattPerceptron *model, const char *filename);
int load_perceptron(RosenblattPerceptron *model, const char *filename);

#endif