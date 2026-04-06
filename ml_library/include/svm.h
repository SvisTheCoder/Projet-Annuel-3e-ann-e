#ifndef SVM_H
#define SVM_H

typedef struct {
    double learning_rate;
    int epochs;
    double lambda;
} SVMClassifier;

void init_svm(SVMClassifier *model, double learning_rate, int epochs, double lambda);
void fit_svm(SVMClassifier *model);
void predict_svm(SVMClassifier *model);
double score_svm(SVMClassifier *model);

#endif