#ifndef ML_API_H
#define ML_API_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct MLModel MLModel;
typedef enum { ML_PERCEPTRON=1, ML_MLP=2, ML_RBF=3, ML_SVM=4 } MLModelType;

typedef struct {
    int epochs;
    double learning_rate;
    int hidden_size;
    int rbf_centers;
    double rbf_sigma;
    double svm_lambda;
    unsigned int seed;
} MLParams;

MLParams ml_default_params(void);
MLModel* ml_create(MLModelType type, int feature_count, int class_count, MLParams params);
int ml_train(MLModel* model, const double* X, const int* y, int sample_count);
int ml_predict(const MLModel* model, const double* x);
double ml_score(const MLModel* model, const double* X, const int* y, int sample_count);
int ml_save(const MLModel* model, const char* path);
MLModel* ml_load(const char* path);
MLModelType ml_type(const MLModel* model);
int ml_feature_count(const MLModel* model);
int ml_class_count(const MLModel* model);
const char* ml_last_error(void);
void ml_free(MLModel* model);
#ifdef __cplusplus
}
#endif
#endif
