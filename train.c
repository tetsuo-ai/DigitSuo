/* ============================================================
 *  ████████▄   ▄█     ▄██████▄   ▄█      ███        ▄████████ ███    █▄   ▄██████▄
 *  ███   ▀███ ███    ███    ███ ███  ▀█████████▄   ███    ███ ███    ███ ███    ███
 *  ███    ███ ███▌   ███    █▀  ███▌    ▀███▀▀██   ███    █▀  ███    ███ ███    ███
 *  ███    ███ ███▌  ▄███        ███▌     ███   ▀   ███        ███    ███ ███    ███
 *  ███    ███ ███▌ ▀▀███ ████▄  ███▌     ███     ▀███████████ ███    ███ ███    ███
 *  ███    ███ ███    ███    ███ ███      ███              ███ ███    ███ ███    ███
 *  ███   ▄███ ███    ███    ███ ███      ███        ▄█    ███ ███    ███ ███    ███
 *  ████████▀  █▀     ████████▀  █▀      ▄████▀    ▄████████▀  ████████▀   ▀██████▀
 *
 *  Project     : DigitSuo
 *  Description : handwritten digit recognition system using a custom
 *                neural network architecture. Built with C, it features both training
 *                capabilities and an interactive recognition interface.
 *                >98% accuracy on the MNIST dataset.
 *  Version     : 1.0
 *  Author      : tetsuo.ai Dev Team :: x.com/7etsuo :: discord.gg/tetsuo-ai
 *  CA          : $Tetsuo on SOLANA  :: 8i51XNNpGaKaj4G4nDdmQh95v4FKAxw8mhtaRoKd9tE8
 *
 *  snowcrash, richinseattle, bobsuo, kokosuo, Petral.S
 *  ============================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <zlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

#define RAND_SEED 42
#define MNIST_TRAIN_SIZE 60000
#define IMAGE_DIM 28
#define IMAGE_CENTER 13.5f
#define GAUSSIAN_SIGMA 0.3f
#define SHIFT_RANGE 5
#define SHIFT_OFFSET 2
#define ROTATION_MAX_DEG 10
#define EPS 1e-10f
#define PRINT_INTERVAL 50
#define PATIENCE 3
#define BASE_LR 0.1f
#define LR_DECAY 0.95f
#define MOMENTUM 0.9f

#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 64
#define NUM_EPOCHS 10
#define SAMPLES_PER_DIGIT 1500
#define TOTAL_SAMPLES (SAMPLES_PER_DIGIT * OUTPUT_SIZE * 2)

typedef struct
{
    float *hidden_weights;
    float *hidden_bias;
    float *output_weights;
    float *output_bias;
    float *hidden_weights_momentum;
    float *hidden_bias_momentum;
    float *output_weights_momentum;
    float *output_bias_momentum;
} Network;

typedef struct
{
    float *batch_X;
    float *batch_y_onehot;
    float *hidden_layer;
    float *output_layer;
    float *hidden_error;
    float *output_error;
    float *dw_hidden;
    float *dw_output;
    float *db_hidden;
    float *db_output;
} TrainingResources;

// clang-format off
float *allocate_array(size_t size);
void initialize_network(Network *net);
void free_network(Network *net);
float random_normal(void);
void read_idx_file(const char *filename, unsigned char *data, int expected_size);
void shuffle_data(unsigned char *images, unsigned char *labels, int n);
float gaussian(float x, float y, float sigma);
void gaussian_filter(float *input, float *output, int size, float sigma);
void rotate_image(unsigned char *input, unsigned char *output, float angle);
void augment_digit(unsigned char *input, unsigned char *output);
void create_augmented_dataset(const unsigned char *train_images, const unsigned char *train_labels,
                              unsigned char *augmented_images, unsigned char *augmented_labels);
float relu(float x);
float relu_derivative(float x);
void softmax(float *input, float *output, int size);
void forward_pass(const Network *net, const float *batch_X, float *hidden_layer, float *output_layer);
void compute_loss_accuracy(const float *output_layer, const float *batch_y_onehot, const unsigned char *labels,
                           int start_idx, float *batch_loss, float *batch_acc);
void backward_pass(const Network *net, const float *batch_X, const float *hidden_layer, const float *output_layer,
                   const float *batch_y_onehot, float *hidden_error, float *output_error, float *dw_hidden,
                   float *dw_output, float *db_hidden, float *db_output);
void update_network(Network *net, const float *dw_hidden, const float *dw_output, const float *db_hidden,
                    const float *db_output, float learning_rate);
void save_weights(Network *net);
// clang-format on

void initialize_training_resources(TrainingResources *res)
{
    res->batch_X = allocate_array(BATCH_SIZE * INPUT_SIZE);
    res->batch_y_onehot = allocate_array(BATCH_SIZE * OUTPUT_SIZE);
    res->hidden_layer = allocate_array(BATCH_SIZE * HIDDEN_SIZE);
    res->output_layer = allocate_array(BATCH_SIZE * OUTPUT_SIZE);
    res->hidden_error = allocate_array(BATCH_SIZE * HIDDEN_SIZE);
    res->output_error = allocate_array(BATCH_SIZE * OUTPUT_SIZE);
    res->dw_hidden = allocate_array(INPUT_SIZE * HIDDEN_SIZE);
    res->dw_output = allocate_array(HIDDEN_SIZE * OUTPUT_SIZE);
    res->db_hidden = allocate_array(HIDDEN_SIZE);
    res->db_output = allocate_array(OUTPUT_SIZE);
}

void free_training_resources(TrainingResources *res)
{
    free(res->batch_X);
    free(res->batch_y_onehot);
    free(res->hidden_layer);
    free(res->output_layer);
    free(res->hidden_error);
    free(res->output_error);
    free(res->dw_hidden);
    free(res->dw_output);
    free(res->db_hidden);
    free(res->db_output);
}

void load_mnist_data(unsigned char **train_images, unsigned char **train_labels)
{
    *train_images = (unsigned char *)malloc(MNIST_TRAIN_SIZE * INPUT_SIZE);
    *train_labels = (unsigned char *)malloc(MNIST_TRAIN_SIZE);
    if (!*train_images || !*train_labels)
    {
        fprintf(stderr, "Memory allocation failed for training data\n");
        exit(1);
    }
    printf("Reading MNIST data...\n");
    read_idx_file("train-images-idx3-ubyte.gz", *train_images, MNIST_TRAIN_SIZE * INPUT_SIZE);
    read_idx_file("train-labels-idx1-ubyte.gz", *train_labels, MNIST_TRAIN_SIZE);
}

void prepare_batch(const unsigned char *images, const unsigned char *labels, int start_idx, float *batch_X,
                   float *batch_y_onehot)
{
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            batch_X[i * INPUT_SIZE + j] = images[(start_idx + i) * INPUT_SIZE + j] / 255.0f;
        }
        memset(&batch_y_onehot[i * OUTPUT_SIZE], 0, OUTPUT_SIZE * sizeof(float));
        batch_y_onehot[i * OUTPUT_SIZE + labels[start_idx + i]] = 1.0f;
    }
}

void train_network(Network *net, unsigned char *aug_images, unsigned char *aug_labels, int total_samples,
                   TrainingResources *res)
{
    int num_batches = total_samples / BATCH_SIZE;
    float best_accuracy = 0.0f;
    int no_improve = 0;

    printf("Starting training...\n");
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++)
    {
        float learning_rate = BASE_LR * powf(LR_DECAY, epoch);
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        shuffle_data(aug_images, aug_labels, total_samples);
        for (int batch = 0; batch < num_batches; batch++)
        {
            int start_idx = batch * BATCH_SIZE;
            prepare_batch(aug_images, aug_labels, start_idx, res->batch_X, res->batch_y_onehot);
            forward_pass(net, res->batch_X, res->hidden_layer, res->output_layer);
            float batch_loss, batch_acc;
            compute_loss_accuracy(res->output_layer, res->batch_y_onehot, aug_labels, start_idx, &batch_loss,
                                  &batch_acc);
            epoch_loss += batch_loss;
            epoch_acc += batch_acc;
            backward_pass(net, res->batch_X, res->hidden_layer, res->output_layer, res->batch_y_onehot,
                          res->hidden_error, res->output_error, res->dw_hidden, res->dw_output, res->db_hidden,
                          res->db_output);
            update_network(net, res->dw_hidden, res->dw_output, res->db_hidden, res->db_output, learning_rate);
            if (batch % PRINT_INTERVAL == 0)
            {
                printf("Batch %d/%d, Loss: %.4f, Accuracy: %.2f%%\n", batch, num_batches, batch_loss,
                       batch_acc * 100.0f);
            }
        }
        epoch_loss /= num_batches;
        epoch_acc /= num_batches;
        printf("Epoch %d/%d, Loss: %.4f, Accuracy: %.2f%%\n", epoch + 1, NUM_EPOCHS, epoch_loss, epoch_acc * 100.0f);
        if (epoch_acc > best_accuracy)
        {
            best_accuracy = epoch_acc;
            no_improve = 0;
            printf("Saving best weights...\n");
            save_weights(net);
        }
        else
        {
            no_improve++;
            if (no_improve >= PATIENCE)
            {
                printf("Early stopping triggered. Best accuracy: %.2f%%\n", best_accuracy * 100.0f);
                break;
            }
        }
    }
    printf("Training completed. Best accuracy: %.2f%%\n", best_accuracy * 100.0f);
}

float *allocate_array(size_t size)
{
    float *array = (float *)malloc(size * sizeof(float));
    if (!array)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    return array;
}

void free_network(Network *net)
{
    free(net->hidden_weights);
    free(net->hidden_bias);
    free(net->output_weights);
    free(net->output_bias);
    free(net->hidden_weights_momentum);
    free(net->hidden_bias_momentum);
    free(net->output_weights_momentum);
    free(net->output_bias_momentum);
}

void initialize_network(Network *net)
{
    net->hidden_weights = allocate_array(INPUT_SIZE * HIDDEN_SIZE);
    net->hidden_bias = allocate_array(HIDDEN_SIZE);
    net->output_weights = allocate_array(HIDDEN_SIZE * OUTPUT_SIZE);
    net->output_bias = allocate_array(OUTPUT_SIZE);
    net->hidden_weights_momentum = allocate_array(INPUT_SIZE * HIDDEN_SIZE);
    net->hidden_bias_momentum = allocate_array(HIDDEN_SIZE);
    net->output_weights_momentum = allocate_array(HIDDEN_SIZE * OUTPUT_SIZE);
    net->output_bias_momentum = allocate_array(OUTPUT_SIZE);
    float scale = sqrtf(2.0f / INPUT_SIZE);
    srand(RAND_SEED);
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
    {
        net->hidden_weights[i] = random_normal() * scale;
        net->hidden_weights_momentum[i] = 0.0f;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
    {
        net->output_weights[i] = random_normal() * scale;
        net->output_weights_momentum[i] = 0.0f;
    }
    memset(net->hidden_bias, 0, HIDDEN_SIZE * sizeof(float));
    memset(net->output_bias, 0, OUTPUT_SIZE * sizeof(float));
    memset(net->hidden_bias_momentum, 0, HIDDEN_SIZE * sizeof(float));
    memset(net->output_bias_momentum, 0, OUTPUT_SIZE * sizeof(float));
}

float random_normal(void)
{
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

void read_idx_file(const char *filename, unsigned char *data, int expected_size)
{
    gzFile file = gzopen(filename, "rb");
    if (!file)
    {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    unsigned char header[16];
    if (gzread(file, header, 4) != 4)
    {
        fprintf(stderr, "Error reading magic number\n");
        exit(1);
    }
    int magic = (header[0] << 24) | (header[1] << 16) | (header[2] << 8) | header[3];
    int dim_count = magic & 0xff;
    int total_size = 1;
    for (int i = 0; i < dim_count; i++)
    {
        if (gzread(file, header + 4 * i, 4) != 4)
        {
            fprintf(stderr, "Error reading dimensions\n");
            exit(1);
        }
        int dim = (header[4 * i] << 24) | (header[4 * i + 1] << 16) | (header[4 * i + 2] << 8) | header[4 * i + 3];
        total_size *= dim;
    }
    if (total_size != expected_size)
    {
        fprintf(stderr, "Unexpected file size\n");
        exit(1);
    }
    if (gzread(file, data, total_size) != total_size)
    {
        fprintf(stderr, "Error reading data\n");
        exit(1);
    }
    gzclose(file);
}

void shuffle_data(unsigned char *images, unsigned char *labels, int n)
{
    for (int i = n - 1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        for (int k = 0; k < INPUT_SIZE; k++)
        {
            unsigned char temp = images[i * INPUT_SIZE + k];
            images[i * INPUT_SIZE + k] = images[j * INPUT_SIZE + k];
            images[j * INPUT_SIZE + k] = temp;
        }
        unsigned char temp = labels[i];
        labels[i] = labels[j];
        labels[j] = temp;
    }
}

float gaussian(float x, float y, float sigma)
{
    float coeff = 1.0f / (2.0f * M_PI * sigma * sigma);
    float expo = -(x * x + y * y) / (2.0f * sigma * sigma);
    return coeff * expf(expo);
}

void gaussian_filter(float *input, float *output, int size, float sigma)
{
    int kernel_size = (int)(6.0f * sigma);
    if (kernel_size % 2 == 0)
        kernel_size++;
    int half = kernel_size / 2;
    float *kernel = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    float sum = 0.0f;
    for (int y = -half; y <= half; y++)
    {
        for (int x = -half; x <= half; x++)
        {
            float g = gaussian((float)x, (float)y, sigma);
            kernel[(y + half) * kernel_size + (x + half)] = g;
            sum += g;
        }
    }
    for (int i = 0; i < kernel_size * kernel_size; i++)
    {
        kernel[i] /= sum;
    }
    for (int y = 0; y < size; y++)
    {
        for (int x = 0; x < size; x++)
        {
            float val = 0.0f;
            for (int ky = -half; ky <= half; ky++)
            {
                for (int kx = -half; kx <= half; kx++)
                {
                    int py = y + ky;
                    int px = x + kx;
                    if (px >= 0 && px < size && py >= 0 && py < size)
                    {
                        val += input[py * size + px] * kernel[(ky + half) * kernel_size + (kx + half)];
                    }
                }
            }
            output[y * size + x] = val;
        }
    }
    free(kernel);
}

void rotate_image(unsigned char *input, unsigned char *output, float angle)
{
    float radian = angle * M_PI / 180.0f;
    float cos_theta = cosf(radian);
    float sin_theta = sinf(radian);
    memset(output, 0, IMAGE_DIM * IMAGE_DIM);
    for (int y = 0; y < IMAGE_DIM; y++)
    {
        for (int x = 0; x < IMAGE_DIM; x++)
        {
            float xc = x - IMAGE_CENTER;
            float yc = y - IMAGE_CENTER;
            float xr = xc * cos_theta - yc * sin_theta + IMAGE_CENTER;
            float yr = xc * sin_theta + yc * cos_theta + IMAGE_CENTER;
            if (xr >= 0 && xr < (IMAGE_DIM - 1) && yr >= 0 && yr < (IMAGE_DIM - 1))
            {
                int x0 = (int)xr;
                int y0 = (int)yr;
                float dx = xr - x0;
                float dy = yr - y0;
                float v00 = input[y0 * IMAGE_DIM + x0];
                float v01 = input[y0 * IMAGE_DIM + (x0 + 1)];
                float v10 = input[(y0 + 1) * IMAGE_DIM + x0];
                float v11 = input[(y0 + 1) * IMAGE_DIM + (x0 + 1)];
                float val = v00 * (1 - dx) * (1 - dy) + v01 * dx * (1 - dy) + v10 * (1 - dx) * dy + v11 * dx * dy;
                output[y * IMAGE_DIM + x] = (unsigned char)val;
            }
        }
    }
}

void augment_digit(unsigned char *input, unsigned char *output)
{
    unsigned char *temp1 = (unsigned char *)malloc(IMAGE_DIM * IMAGE_DIM);
    unsigned char *temp2 = (unsigned char *)malloc(IMAGE_DIM * IMAGE_DIM);
    float *float_buffer1 = (float *)malloc(IMAGE_DIM * IMAGE_DIM * sizeof(float));
    float *float_buffer2 = (float *)malloc(IMAGE_DIM * IMAGE_DIM * sizeof(float));
    if (!temp1 || !temp2 || !float_buffer1 || !float_buffer2)
    {
        fprintf(stderr, "Failed to allocate memory for augmentation\n");
        exit(1);
    }
    float angle = ((float)rand() / RAND_MAX) * (2.0f * ROTATION_MAX_DEG) - ROTATION_MAX_DEG;
    rotate_image(input, temp1, angle);
    int shift_x = rand() % SHIFT_RANGE - SHIFT_OFFSET;
    int shift_y = rand() % SHIFT_RANGE - SHIFT_OFFSET;
    memset(temp2, 0, IMAGE_DIM * IMAGE_DIM);
    for (int y = 0; y < IMAGE_DIM; y++)
    {
        for (int x = 0; x < IMAGE_DIM; x++)
        {
            int new_x = x + shift_x;
            int new_y = y + shift_y;
            if (new_x >= 0 && new_x < IMAGE_DIM && new_y >= 0 && new_y < IMAGE_DIM)
            {
                temp2[new_y * IMAGE_DIM + new_x] = temp1[y * IMAGE_DIM + x];
            }
        }
    }
    for (int i = 0; i < IMAGE_DIM * IMAGE_DIM; i++)
    {
        float_buffer1[i] = temp2[i] / 255.0f;
    }
    gaussian_filter(float_buffer1, float_buffer2, IMAGE_DIM, GAUSSIAN_SIGMA);
    for (int i = 0; i < IMAGE_DIM * IMAGE_DIM; i++)
    {
        float val = float_buffer2[i] * 255.0f;
        output[i] = (unsigned char)((val < 0) ? 0 : ((val > 255) ? 255 : val));
    }
    free(temp1);
    free(temp2);
    free(float_buffer1);
    free(float_buffer2);
}

void create_augmented_dataset(const unsigned char *train_images, const unsigned char *train_labels,
                              unsigned char *augmented_images, unsigned char *augmented_labels)
{
    int *digit_counts = (int *)calloc(OUTPUT_SIZE, sizeof(int));
    int **digit_indices = (int **)malloc(OUTPUT_SIZE * sizeof(int *));
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        digit_indices[i] = (int *)malloc(MNIST_TRAIN_SIZE * sizeof(int));
        digit_counts[i] = 0;
    }
    for (int i = 0; i < MNIST_TRAIN_SIZE; i++)
    {
        int digit = train_labels[i];
        digit_indices[digit][digit_counts[digit]++] = i;
    }
    int sample_idx = 0;
    for (int digit = 0; digit < OUTPUT_SIZE; digit++)
    {
        for (int j = 0; j < SAMPLES_PER_DIGIT; j++)
        {
            int remain = digit_counts[digit] - j;
            int rand_idx = j + (rand() % remain);
            int tmp_idx = digit_indices[digit][j];
            digit_indices[digit][j] = digit_indices[digit][rand_idx];
            digit_indices[digit][rand_idx] = tmp_idx;
            int idx = digit_indices[digit][j];
            memcpy(&augmented_images[sample_idx * INPUT_SIZE], &train_images[idx * INPUT_SIZE], INPUT_SIZE);
            augmented_labels[sample_idx] = digit;
            sample_idx++;
            augment_digit((unsigned char *)&train_images[idx * INPUT_SIZE], &augmented_images[sample_idx * INPUT_SIZE]);
            augmented_labels[sample_idx] = digit;
            sample_idx++;
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        free(digit_indices[i]);
    }
    free(digit_indices);
    free(digit_counts);
}

float relu(float x)
{
    return (x > 0) ? x : 0;
}

float relu_derivative(float x)
{
    return (x > 0) ? 1.0f : 0.0f;
}

void softmax(float *input, float *output, int size)
{
    float max_val = input[0];
    for (int i = 1; i < size; i++)
    {
        if (input[i] > max_val)
            max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++)
    {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++)
    {
        output[i] /= sum;
    }
}

void forward_pass(const Network *net, const float *batch_X, float *hidden_layer, float *output_layer)
{
#pragma omp parallel for collapse(2)
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            float sum = net->hidden_bias[j];
            for (int k = 0; k < INPUT_SIZE; k++)
            {
                sum += batch_X[i * INPUT_SIZE + k] * net->hidden_weights[k * HIDDEN_SIZE + j];
            }
            hidden_layer[i * HIDDEN_SIZE + j] = relu(sum);
        }
    }

#pragma omp parallel for
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        float tmp[OUTPUT_SIZE];
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            float sum = net->output_bias[j];
            for (int k = 0; k < HIDDEN_SIZE; k++)
            {
                sum += hidden_layer[i * HIDDEN_SIZE + k] * net->output_weights[k * OUTPUT_SIZE + j];
            }
            tmp[j] = sum;
        }
        softmax(tmp, &output_layer[i * OUTPUT_SIZE], OUTPUT_SIZE);
    }
}

void compute_loss_accuracy(const float *output_layer, const float *batch_y_onehot, const unsigned char *labels,
                           int start_idx, float *batch_loss, float *batch_acc)
{
    float loss_val = 0.0f;
    int correct = 0;
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        float single_loss = 0.0f;
        float max_prob = output_layer[i * OUTPUT_SIZE];
        int predicted = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            float prob = output_layer[i * OUTPUT_SIZE + j];
            if (prob > max_prob)
            {
                max_prob = prob;
                predicted = j;
            }
            if (batch_y_onehot[i * OUTPUT_SIZE + j] > 0.5f)
            {
                single_loss -= logf(prob + EPS);
            }
        }
        if (predicted == labels[start_idx + i])
        {
            correct++;
        }
        loss_val += single_loss;
    }
    *batch_loss = loss_val / BATCH_SIZE;
    *batch_acc = (float)correct / BATCH_SIZE;
}

void backward_pass(const Network *net, const float *batch_X, const float *hidden_layer, const float *output_layer,
                   const float *batch_y_onehot, float *hidden_error, float *output_error, float *dw_hidden,
                   float *dw_output, float *db_hidden, float *db_output)
{
#pragma omp parallel sections
    {
#pragma omp section
        memset(dw_hidden, 0, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
#pragma omp section
        memset(dw_output, 0, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
#pragma omp section
        memset(db_hidden, 0, HIDDEN_SIZE * sizeof(float));
#pragma omp section
        memset(db_output, 0, OUTPUT_SIZE * sizeof(float));
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            output_error[i * OUTPUT_SIZE + j] = output_layer[i * OUTPUT_SIZE + j] - batch_y_onehot[i * OUTPUT_SIZE + j];
        }
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < BATCH_SIZE; i++)
    {
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            float sum_err = 0.0f;
            for (int k = 0; k < OUTPUT_SIZE; k++)
            {
                sum_err += output_error[i * OUTPUT_SIZE + k] * net->output_weights[j * OUTPUT_SIZE + k];
            }
            hidden_error[i * HIDDEN_SIZE + j] = sum_err * relu_derivative(hidden_layer[i * HIDDEN_SIZE + j]);
        }
    }

#pragma omp parallel
    {
#pragma omp for collapse(2)
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            for (int k = 0; k < HIDDEN_SIZE; k++)
            {
                float grad = 0.0f;
                for (int i = 0; i < BATCH_SIZE; i++)
                {
                    grad += batch_X[i * INPUT_SIZE + j] * hidden_error[i * HIDDEN_SIZE + k];
                }
                dw_hidden[j * HIDDEN_SIZE + k] = grad / BATCH_SIZE;
            }
        }

#pragma omp for collapse(2)
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            for (int k = 0; k < OUTPUT_SIZE; k++)
            {
                float grad = 0.0f;
                for (int i = 0; i < BATCH_SIZE; i++)
                {
                    grad += hidden_layer[i * HIDDEN_SIZE + j] * output_error[i * OUTPUT_SIZE + k];
                }
                dw_output[j * OUTPUT_SIZE + k] = grad / BATCH_SIZE;
            }
        }

#pragma omp for
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            float grad = 0.0f;
            for (int i = 0; i < BATCH_SIZE; i++)
            {
                grad += hidden_error[i * HIDDEN_SIZE + j];
            }
            db_hidden[j] = grad / BATCH_SIZE;
        }

#pragma omp for
        for (int j = 0; j < OUTPUT_SIZE; j++)
        {
            float grad = 0.0f;
            for (int i = 0; i < BATCH_SIZE; i++)
            {
                grad += output_error[i * OUTPUT_SIZE + j];
            }
            db_output[j] = grad / BATCH_SIZE;
        }
    }
}

void update_network(Network *net, const float *dw_hidden, const float *dw_output, const float *db_hidden,
                    const float *db_output, float learning_rate)
{
#pragma omp parallel sections
    {
#pragma omp section
        {
            for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
            {
                net->hidden_weights_momentum[i] =
                    MOMENTUM * net->hidden_weights_momentum[i] - learning_rate * dw_hidden[i];
                net->hidden_weights[i] += net->hidden_weights_momentum[i];
            }
        }
#pragma omp section
        {
            for (int i = 0; i < HIDDEN_SIZE; i++)
            {
                net->hidden_bias_momentum[i] = MOMENTUM * net->hidden_bias_momentum[i] - learning_rate * db_hidden[i];
                net->hidden_bias[i] += net->hidden_bias_momentum[i];
            }
        }
#pragma omp section
        {
            for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
            {
                net->output_weights_momentum[i] =
                    MOMENTUM * net->output_weights_momentum[i] - learning_rate * dw_output[i];
                net->output_weights[i] += net->output_weights_momentum[i];
            }
        }
#pragma omp section
        {
            for (int i = 0; i < OUTPUT_SIZE; i++)
            {
                net->output_bias_momentum[i] = MOMENTUM * net->output_bias_momentum[i] - learning_rate * db_output[i];
                net->output_bias[i] += net->output_bias_momentum[i];
            }
        }
    }
}

void save_weights(Network *net)
{
    FILE *f = fopen("src/weights.h", "w");
    if (!f)
    {
        fprintf(stderr, "Error opening weights.h for writing\n");
        return;
    }
    fprintf(f, "#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n");
    fprintf(f, "#define INPUT_SIZE %d\n", INPUT_SIZE);
    fprintf(f, "#define HIDDEN_SIZE %d\n", HIDDEN_SIZE);
    fprintf(f, "#define OUTPUT_SIZE %d\n\n", OUTPUT_SIZE);
    fprintf(f, "static const float HIDDEN_WEIGHTS[INPUT_SIZE * HIDDEN_SIZE] = {\n");
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++)
    {
        fprintf(f, "    %10.6ff%s", net->hidden_weights[i], (i + 1 < INPUT_SIZE * HIDDEN_SIZE) ? "," : "");
        if ((i + 1) % 8 == 0)
            fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
    fprintf(f, "static const float HIDDEN_BIAS[HIDDEN_SIZE] = {\n");
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        fprintf(f, "    %10.6ff%s", net->hidden_bias[i], (i + 1 < HIDDEN_SIZE) ? "," : "");
        if ((i + 1) % 8 == 0)
            fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
    fprintf(f, "static const float OUTPUT_WEIGHTS[HIDDEN_SIZE * OUTPUT_SIZE] = {\n");
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++)
    {
        fprintf(f, "    %10.6ff%s", net->output_weights[i], (i + 1 < HIDDEN_SIZE * OUTPUT_SIZE) ? "," : "");
        if ((i + 1) % 8 == 0)
            fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
    fprintf(f, "static const float OUTPUT_BIAS[OUTPUT_SIZE] = {\n");
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        fprintf(f, "    %10.6ff%s", net->output_bias[i], (i + 1 < OUTPUT_SIZE) ? "," : "");
        if ((i + 1) % 8 == 0)
            fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
    fprintf(f, "#endif /* WEIGHTS_H */\n");
    fclose(f);
    printf("Successfully saved weights to src/weights.h\n");
}

int main(void)
{
    srand(RAND_SEED);
    Network net;
    initialize_network(&net);
    unsigned char *train_images, *train_labels;
    load_mnist_data(&train_images, &train_labels);
    unsigned char *aug_images = (unsigned char *)malloc(TOTAL_SAMPLES * INPUT_SIZE);
    unsigned char *aug_labels = (unsigned char *)malloc(TOTAL_SAMPLES);
    if (!aug_images || !aug_labels)
    {
        fprintf(stderr, "Failed to allocate memory for augmented dataset\n");
        return 1;
    }
    create_augmented_dataset(train_images, train_labels, aug_images, aug_labels);
    TrainingResources res;
    initialize_training_resources(&res);
    train_network(&net, aug_images, aug_labels, TOTAL_SAMPLES, &res);
    free_training_resources(&res);
    free(aug_images);
    free(aug_labels);
    free(train_images);
    free(train_labels);
    free_network(&net);
    return 0;
}
