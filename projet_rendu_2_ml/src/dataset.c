#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int count_csv_values(const char* line) {
    int count = 1;

    for (const char* cursor = line; *cursor != '\0'; cursor++) {
        if (*cursor == ',') {
            count++;
        }
    }

    return count;
}

int dataset_load_csv(const char* path, Dataset* output) {
    FILE* file;
    char line[65536];
    int row_count = 0;
    int column_count = 0;

    if (path == NULL || output == NULL) {
        return 0;
    }

    memset(output, 0, sizeof(*output));
    file = fopen(path, "r");

    if (file == NULL) {
        return 0;
    }

    /* Premier passage : compter les lignes et les colonnes. */
    while (fgets(line, sizeof(line), file) != NULL) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }

        if (column_count == 0) {
            column_count = count_csv_values(line);
        }

        row_count++;
    }

    if (row_count == 0 || column_count < 2) {
        fclose(file);
        return 0;
    }

    rewind(file);

    output->sample_count = row_count;
    output->feature_count = column_count - 1;
    output->X = malloc(
        sizeof(double) * row_count * output->feature_count
    );
    output->y = malloc(sizeof(int) * row_count);

    if (output->X == NULL || output->y == NULL) {
        dataset_free(output);
        fclose(file);
        return 0;
    }

    /* Deuxieme passage : convertir les valeurs du CSV. */
    int row = 0;
    int maximum_label = -1;

    while (fgets(line, sizeof(line), file) != NULL) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') {
            continue;
        }

        char* token = strtok(line, ",");

        for (int column = 0; column < column_count; column++) {
            if (token == NULL) {
                dataset_free(output);
                fclose(file);
                return 0;
            }

            if (column < output->feature_count) {
                output->X[row * output->feature_count + column] =
                    strtod(token, NULL);
            } else {
                output->y[row] = (int)strtol(token, NULL, 10);

                if (output->y[row] < 0) {
                    dataset_free(output);
                    fclose(file);
                    return 0;
                }

                if (output->y[row] > maximum_label) {
                    maximum_label = output->y[row];
                }
            }

            token = strtok(NULL, ",");
        }

        row++;
    }

    fclose(file);
    output->class_count = maximum_label + 1;
    return 1;
}

int dataset_save_csv(const char* path, const Dataset* data) {
    if (path == NULL || data == NULL) {
        return 0;
    }

    FILE* file = fopen(path, "w");

    if (file == NULL) {
        return 0;
    }

    for (int sample = 0; sample < data->sample_count; sample++) {
        for (int feature = 0;
             feature < data->feature_count;
             feature++) {
            fprintf(
                file,
                "%.10g,",
                data->X[sample * data->feature_count + feature]
            );
        }

        fprintf(file, "%d\n", data->y[sample]);
    }

    fclose(file);
    return 1;
}

void dataset_free(Dataset* data) {
    if (data == NULL) {
        return;
    }

    free(data->X);
    free(data->y);
    memset(data, 0, sizeof(*data));
}

static unsigned int next_random(unsigned int* state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static int allocate_dataset_like(
    const Dataset* source,
    int sample_count,
    Dataset* output
) {
    memset(output, 0, sizeof(*output));
    output->sample_count = sample_count;
    output->feature_count = source->feature_count;
    output->class_count = source->class_count;
    output->X = malloc(
        sizeof(double) * sample_count * source->feature_count
    );
    output->y = malloc(sizeof(int) * sample_count);

    if (output->X == NULL || output->y == NULL) {
        dataset_free(output);
        return 0;
    }

    return 1;
}

int dataset_split(
    const Dataset* source,
    double test_ratio,
    unsigned int seed,
    Dataset* train,
    Dataset* test
) {
    if (source == NULL
        || train == NULL
        || test == NULL
        || source->sample_count < 2
        || test_ratio <= 0.0
        || test_ratio >= 1.0) {
        return 0;
    }

    int test_count = (int)(source->sample_count * test_ratio);

    if (test_count < 1) {
        test_count = 1;
    }

    int train_count = source->sample_count - test_count;
    int* indices = malloc(sizeof(int) * source->sample_count);

    if (indices == NULL) {
        return 0;
    }

    for (int i = 0; i < source->sample_count; i++) {
        indices[i] = i;
    }

    /* Melange de Fisher-Yates avec une graine reproductible. */
    for (int i = source->sample_count - 1; i > 0; i--) {
        int random_index = (int)(next_random(&seed) % (unsigned int)(i + 1));
        int temporary = indices[i];
        indices[i] = indices[random_index];
        indices[random_index] = temporary;
    }

    if (!allocate_dataset_like(source, train_count, train)
        || !allocate_dataset_like(source, test_count, test)) {
        free(indices);
        dataset_free(train);
        dataset_free(test);
        return 0;
    }

    for (int sample = 0; sample < train_count; sample++) {
        int source_index = indices[sample];

        memcpy(
            &train->X[sample * source->feature_count],
            &source->X[source_index * source->feature_count],
            sizeof(double) * source->feature_count
        );
        train->y[sample] = source->y[source_index];
    }

    for (int sample = 0; sample < test_count; sample++) {
        int source_index = indices[train_count + sample];

        memcpy(
            &test->X[sample * source->feature_count],
            &source->X[source_index * source->feature_count],
            sizeof(double) * source->feature_count
        );
        test->y[sample] = source->y[source_index];
    }

    free(indices);
    return 1;
}

void dataset_print_info(const Dataset* data) {
    if (data == NULL) {
        return;
    }

    printf("Exemples : %d\n", data->sample_count);
    printf("Caracteristiques : %d\n", data->feature_count);
    printf("Classes : %d\n", data->class_count);
}
