#ifndef DATASET_H
#define DATASET_H

typedef struct {
    double* X;
    int* y;
    int sample_count;
    int feature_count;
    int class_count;
} Dataset;

int dataset_load_csv(const char* path, Dataset* out);
int dataset_save_csv(const char* path, const Dataset* data);
void dataset_free(Dataset* data);
int dataset_split(const Dataset* src, double test_ratio, unsigned int seed, Dataset* train, Dataset* test);
void dataset_print_info(const Dataset* data);
#endif
