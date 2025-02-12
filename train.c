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
#include <immintrin.h>    // We use these fancy AVX tricks for number crunching
#include <cblas.h>        // BLAS helps us do matrix math super fast, this is new for enhanced optimization, like insanely faster than the other one fr fr, like i aint even capping

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef _OPENMP
#define omp_get_thread_num() 0
#define omp_get_max_threads() 1
#endif

// These are our settings—think of them as our recipe's secret ingredients.
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

// More settings for our network—this is how we shape our brain.
#define INPUT_SIZE 784
#define HIDDEN_SIZE 256
#define OUTPUT_SIZE 10
#define BATCH_SIZE 64
#define NUM_EPOCHS 10
#define SAMPLES_PER_DIGIT 1500
#define TOTAL_SAMPLES (SAMPLES_PER_DIGIT * OUTPUT_SIZE * 2)

// This little function sums up 8 floats from an AVX register.
// It’s like taking 8 ingredients and mixing them together into one tasty number.
static inline float hsum256_ps(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);      // Grab the lower half
    __m128 vhigh = _mm256_extractf128_ps(v, 1);      // And the upper half too
    vlow = _mm_add_ps(vlow, vhigh);                  // Mix them up together
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);                      // And return our yummy sum!
}

// This structure holds all the juicy details of our network—weights, biases, and momentum data.
typedef struct {
    float *hidden_weights;
    float *hidden_bias;
    float *output_weights;
    float *output_bias;
    float *hidden_weights_momentum;
    float *hidden_bias_momentum;
    float *output_weights_momentum;
    float *output_bias_momentum;
} Network;

// These are our working buffers for training. Think of them as our scratch paper.
typedef struct {
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

/////////////////////////////////////////////
// Super simple memory allocation helper.
// It makes sure we get our memory or it just crashes out.
/////////////////////////////////////////////
float *allocate_array(size_t size) {
    float *array = (float *)malloc(size * sizeof(float));
    if (!array) {
        fprintf(stderr, "Uh oh, ran out of memory!\n");
        exit(1);
    }
    return array;
}

/////////////////////////////////////////////
// This function spits out a random number following a normal distribution.
// We use it to give our network a nice random start.
/////////////////////////////////////////////
float random_normal(void) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

/////////////////////////////////////////////
// Time to set up our network. We allocate all the weights and biases
// and then give them some random values so our network isn’t biased from the get-go.
/////////////////////////////////////////////
void initialize_network(Network *net) {
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
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        net->hidden_weights[i] = random_normal() * scale;
        net->hidden_weights_momentum[i] = 0.0f;
    }
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        net->output_weights[i] = random_normal() * scale;
        net->output_weights_momentum[i] = 0.0f;
    }
    memset(net->hidden_bias, 0, HIDDEN_SIZE * sizeof(float));
    memset(net->output_bias, 0, OUTPUT_SIZE * sizeof(float));
    memset(net->hidden_bias_momentum, 0, HIDDEN_SIZE * sizeof(float));
    memset(net->output_bias_momentum, 0, OUTPUT_SIZE * sizeof(float));
}

/////////////////////////////////////////////
// This function reads an IDX file (that's the format for MNIST) and loads its data.
// It’s our doorway to all the handwritten digits.
/////////////////////////////////////////////
void read_idx_file(const char *filename, unsigned char *data, int expected_size) {
    gzFile file = gzopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Couldn’t open file: %s\n", filename);
        exit(1);
    }
    unsigned char header[16];
    if (gzread(file, header, 4) != 4) {
        fprintf(stderr, "Hmm, can’t read the magic number.\n");
        exit(1);
    }
    int magic = (header[0] << 24) | (header[1] << 16) | (header[2] << 8) | header[3];
    int dim_count = magic & 0xff;
    int total_size = 1;
    for (int i = 0; i < dim_count; i++) {
        if (gzread(file, header + 4 * i, 4) != 4) {
            fprintf(stderr, "Error reading dimensions – something’s not right!\n");
            exit(1);
        }
        int dim = (header[4 * i] << 24) | (header[4 * i + 1] << 16) | (header[4 * i + 2] << 8) | header[4 * i + 3];
        total_size *= dim;
    }
    if (total_size != expected_size) {
        fprintf(stderr, "Whoa, unexpected file size!\n");
        exit(1);
    }
    if (gzread(file, data, total_size) != total_size) {
        fprintf(stderr, "Error reading the actual data – uh oh!\n");
        exit(1);
    }
    gzclose(file);
}

/////////////////////////////////////////////
// Shuffle your dataset so it’s all mixed up – like shuffling a deck of cards.
/////////////////////////////////////////////
void shuffle_data(unsigned char *images, unsigned char *labels, int n) {
    #pragma omp parallel for
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        for (int k = 0; k < INPUT_SIZE; k++) {
            unsigned char temp = images[i * INPUT_SIZE + k];
            images[i * INPUT_SIZE + k] = images[j * INPUT_SIZE + k];
            images[j * INPUT_SIZE + k] = temp;
        }
        unsigned char temp = labels[i];
        labels[i] = labels[j];
        labels[j] = temp;
    }
}

/////////////////////////////////////////////
// This part messes with the images to create more training examples.
// We rotate, shift, and smooth the images so the network sees lots of variations.
/////////////////////////////////////////////
float gaussian(float x, float y, float sigma) {
    float coeff = 1.0f / (2.0f * M_PI * sigma * sigma);
    float expo = -(x * x + y * y) / (2.0f * sigma * sigma);
    return coeff * expf(expo);
}

void gaussian_filter(float *input, float *output, int size, float sigma) {
    int kernel_size = (int)(6.0f * sigma);
    if (kernel_size % 2 == 0)
        kernel_size++;
    int half = kernel_size / 2;
    float *kernel = (float *)malloc(kernel_size * kernel_size * sizeof(float));
    float sum = 0.0f;
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float g = gaussian((float)x, (float)y, sigma);
            kernel[(y + half) * kernel_size + (x + half)] = g;
            sum += g;
        }
    }
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        kernel[i] /= sum;
    }
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < size; y++) {
        for (int x = 0; x < size; x++) {
            float val = 0.0f;
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    int py = y + ky;
                    int px = x + kx;
                    if (px >= 0 && px < size && py >= 0 && py < size) {
                        val += input[py * size + px] * kernel[(ky + half) * kernel_size + (kx + half)];
                    }
                }
            }
            output[y * size + x] = val;
        }
    }
    free(kernel);
}

void rotate_image(unsigned char *input, unsigned char *output, float angle) {
    float radian = angle * M_PI / 180.0f;
    float cos_theta = cosf(radian);
    float sin_theta = sinf(radian);
    memset(output, 0, IMAGE_DIM * IMAGE_DIM);
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < IMAGE_DIM; y++) {
        for (int x = 0; x < IMAGE_DIM; x++) {
            float xc = x - IMAGE_CENTER;
            float yc = y - IMAGE_CENTER;
            float xr = xc * cos_theta - yc * sin_theta + IMAGE_CENTER;
            float yr = xc * sin_theta + yc * cos_theta + IMAGE_CENTER;
            if (xr >= 0 && xr < (IMAGE_DIM - 1) && yr >= 0 && yr < (IMAGE_DIM - 1)) {
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

void augment_digit(unsigned char *input, unsigned char *output) {
    unsigned char *temp1 = (unsigned char *)malloc(IMAGE_DIM * IMAGE_DIM);
    unsigned char *temp2 = (unsigned char *)malloc(IMAGE_DIM * IMAGE_DIM);
    float *float_buffer1 = (float *)malloc(IMAGE_DIM * IMAGE_DIM * sizeof(float));
    float *float_buffer2 = (float *)malloc(IMAGE_DIM * IMAGE_DIM * sizeof(float));
    if (!temp1 || !temp2 || !float_buffer1 || !float_buffer2) {
        fprintf(stderr, "Oof, couldn’t get memory for image augmentation!\n");
        exit(1);
    }
    // Give the image a random spin
    float angle = ((float)rand() / RAND_MAX) * (2.0f * ROTATION_MAX_DEG) - ROTATION_MAX_DEG;
    rotate_image(input, temp1, angle);
    // Also shift it a little
    int shift_x = rand() % SHIFT_RANGE - SHIFT_OFFSET;
    int shift_y = rand() % SHIFT_RANGE - SHIFT_OFFSET;
    memset(temp2, 0, IMAGE_DIM * IMAGE_DIM);
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < IMAGE_DIM; y++) {
        for (int x = 0; x < IMAGE_DIM; x++) {
            int new_x = x + shift_x;
            int new_y = y + shift_y;
            if (new_x >= 0 && new_x < IMAGE_DIM && new_y >= 0 && new_y < IMAGE_DIM) {
                temp2[new_y * IMAGE_DIM + new_x] = temp1[y * IMAGE_DIM + x];
            }
        }
    }
    #pragma omp parallel for
    for (int i = 0; i < IMAGE_DIM * IMAGE_DIM; i++) {
        float_buffer1[i] = temp2[i] / 255.0f;
    }
    gaussian_filter(float_buffer1, float_buffer2, IMAGE_DIM, GAUSSIAN_SIGMA);
    #pragma omp parallel for
    for (int i = 0; i < IMAGE_DIM * IMAGE_DIM; i++) {
        float val = float_buffer2[i] * 255.0f;
        output[i] = (unsigned char)((val < 0) ? 0 : ((val > 255) ? 255 : val));
    }
    free(temp1);
    free(temp2);
    free(float_buffer1);
    free(float_buffer2);
}

void create_augmented_dataset(const unsigned char *train_images, const unsigned char *train_labels,
                              unsigned char *augmented_images, unsigned char *augmented_labels) {
    int *digit_counts = (int *)calloc(OUTPUT_SIZE, sizeof(int));
    int **digit_indices = (int **)malloc(OUTPUT_SIZE * sizeof(int *));
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        digit_indices[i] = (int *)malloc(MNIST_TRAIN_SIZE * sizeof(int));
        digit_counts[i] = 0;
    }
    // Tally up how many images we have for each digit
    for (int i = 0; i < MNIST_TRAIN_SIZE; i++) {
        int digit = train_labels[i];
        digit_indices[digit][digit_counts[digit]++] = i;
    }
    int sample_idx = 0;
    // For each digit, pick some random images and then augment them
    for (int digit = 0; digit < OUTPUT_SIZE; digit++) {
        for (int j = 0; j < SAMPLES_PER_DIGIT; j++) {
            int remain = digit_counts[digit] - j;
            int rand_idx = j + (rand() % remain);
            int tmp_idx = digit_indices[digit][j];
            digit_indices[digit][j] = digit_indices[digit][rand_idx];
            digit_indices[digit][rand_idx] = tmp_idx;
            int idx = digit_indices[digit][j];
            memcpy(&augmented_images[sample_idx * INPUT_SIZE], &train_images[idx * INPUT_SIZE], INPUT_SIZE);
            augmented_labels[sample_idx] = digit;
            sample_idx++;
            augment_digit((unsigned char *)&train_images[idx * INPUT_SIZE],
                          &augmented_images[sample_idx * INPUT_SIZE]);
            augmented_labels[sample_idx] = digit;
            sample_idx++;
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        free(digit_indices[i]);
    }
    free(digit_indices);
    free(digit_counts);
}

/////////////////////////////////////////////
// Here we define our activation functions in a very straightforward way.
// relu just gives you the value if it’s positive, or zero otherwise.
/////////////////////////////////////////////
float relu(float x) {
    return (x > 0) ? x : 0;
}

float relu_derivative(float x) {
    return (x > 0) ? 1.0f : 0.0f;
}

/////////////////////////////////////////////
// Softmax turns raw output scores into probabilities.
// It’s like turning your exam scores into percentages.
/////////////////////////////////////////////
void softmax(float *input, float *output, int size) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val)
            max_val = input[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

/////////////////////////////////////////////
// This is our forward pass – the part where the network makes a guess.
// We use BLAS to speed up our matrix multiplications.
/////////////////////////////////////////////
void forward_pass(const Network *net, const float *batch_X, float *hidden_layer, float *output_layer) {
    // Multiply the input by the hidden weights
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                BATCH_SIZE, HIDDEN_SIZE, INPUT_SIZE, 1.0f,
                batch_X, INPUT_SIZE, net->hidden_weights, HIDDEN_SIZE,
                0.0f, hidden_layer, HIDDEN_SIZE);
    // Add in the hidden biases (each sample gets the same bias)
    #pragma omp parallel for
    for (int i = 0; i < BATCH_SIZE; i++) {
        cblas_saxpy(HIDDEN_SIZE, 1.0f, net->hidden_bias, 1, &hidden_layer[i * HIDDEN_SIZE], 1);
    }
    // Run ReLU on the hidden layer so we only keep positive values
    #pragma omp parallel for
    for (int i = 0; i < BATCH_SIZE * HIDDEN_SIZE; i++) {
        if (hidden_layer[i] < 0) hidden_layer[i] = 0;
    }
    // Now compute the output layer by multiplying the hidden layer with the output weights
    float *temp_output = (float *)malloc(BATCH_SIZE * OUTPUT_SIZE * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, 1.0f,
                hidden_layer, HIDDEN_SIZE, net->output_weights, OUTPUT_SIZE,
                0.0f, temp_output, OUTPUT_SIZE);
    // Add the output biases
    #pragma omp parallel for
    for (int i = 0; i < BATCH_SIZE; i++) {
        cblas_saxpy(OUTPUT_SIZE, 1.0f, net->output_bias, 1, &temp_output[i * OUTPUT_SIZE], 1);
    }
    // Finally, run softmax on each sample to get probability distributions
    #pragma omp parallel for
    for (int i = 0; i < BATCH_SIZE; i++) {
        softmax(&temp_output[i * OUTPUT_SIZE], &output_layer[i * OUTPUT_SIZE], OUTPUT_SIZE);
    }
    free(temp_output);
}

/////////////////////////////////////////////
// Here we calculate how off our guesses are and how many we got right.
// It’s our way of checking our network’s homework.
/////////////////////////////////////////////
void compute_loss_accuracy(const float *output_layer, const float *batch_y_onehot, const unsigned char *labels,
                           int start_idx, float *batch_loss, float *batch_acc) {
    float loss_val = 0.0f;
    int correct = 0;
    #pragma omp parallel for reduction(+:loss_val,correct)
    for (int i = 0; i < BATCH_SIZE; i++) {
        float single_loss = 0.0f;
        float max_prob = output_layer[i * OUTPUT_SIZE];
        int predicted = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            float prob = output_layer[i * OUTPUT_SIZE + j];
            if (prob > max_prob) {
                max_prob = prob;
                predicted = j;
            }
            if (batch_y_onehot[i * OUTPUT_SIZE + j] > 0.5f) {
                single_loss -= logf(prob + EPS);
            }
        }
        if (predicted == labels[start_idx + i]) {
            correct++;
        }
        loss_val += single_loss;
    }
    *batch_loss = loss_val / BATCH_SIZE;
    *batch_acc = (float)correct / BATCH_SIZE;
}

/////////////////////////////////////////////
// This is where the magic of backprop happens.
// We calculate how much each weight contributed to our error.
/////////////////////////////////////////////
void backward_pass(const Network *net, const float *batch_X, const float *hidden_layer, const float *output_layer,
                   const float *batch_y_onehot, float *hidden_error, float *output_error, float *dw_hidden,
                   float *dw_output, float *db_hidden, float *db_output) {
    int i, j, k;
    // First, figure out the error at the output (our guess minus the truth)
    int total_output = BATCH_SIZE * OUTPUT_SIZE;
    #pragma omp parallel for
    for (i = 0; i < total_output; i++) {
        output_error[i] = output_layer[i] - batch_y_onehot[i];
    }
    // Now, calculate the gradient for the output weights (hidden_layer^T * output_error)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE, 1.0f,
                hidden_layer, HIDDEN_SIZE, output_error, OUTPUT_SIZE,
                0.0f, dw_output, OUTPUT_SIZE);
    #pragma omp parallel for
    for (i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        dw_output[i] /= BATCH_SIZE;
    }
    // Sum the errors for the output biases
    #pragma omp parallel for
    for (j = 0; j < OUTPUT_SIZE; j++) {
        float sum = 0.0f;
        for (i = 0; i < BATCH_SIZE; i++) {
            sum += output_error[i * OUTPUT_SIZE + j];
        }
        db_output[j] = sum / BATCH_SIZE;
    }
    // Propagate the error back into the hidden layer using output weights
    float *temp_hidden_error = (float *)malloc(BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, 1.0f,
                output_error, OUTPUT_SIZE, net->output_weights, OUTPUT_SIZE,
                0.0f, temp_hidden_error, HIDDEN_SIZE);
    // Apply the derivative of ReLU to get the hidden error
    #pragma omp parallel for
    for (i = 0; i < BATCH_SIZE * HIDDEN_SIZE; i++) {
        float derivative = (hidden_layer[i] > 0) ? 1.0f : 0.0f;
        hidden_error[i] = temp_hidden_error[i] * derivative;
    }
    free(temp_hidden_error);
    // Now compute the gradient for the hidden weights
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                INPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE, 1.0f,
                batch_X, INPUT_SIZE, hidden_error, HIDDEN_SIZE,
                0.0f, dw_hidden, HIDDEN_SIZE);
    #pragma omp parallel for
    for (i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        dw_hidden[i] /= BATCH_SIZE;
    }
    // And sum up the hidden errors for the hidden biases
    #pragma omp parallel for
    for (j = 0; j < HIDDEN_SIZE; j++) {
        float sum = 0.0f;
        for (i = 0; i < BATCH_SIZE; i++) {
            sum += hidden_error[i * HIDDEN_SIZE + j];
        }
        db_hidden[j] = sum / BATCH_SIZE;
    }
}

/////////////////////////////////////////////
// Here we update our network parameters using the gradients we calculated.
// We use momentum here to help smooth out the updates, and we lean on BLAS routines.
/////////////////////////////////////////////
void update_network(Network *net, const float *dw_hidden, const float *dw_output, const float *db_hidden,
                    const float *db_output, float learning_rate) {
    int size_hw = INPUT_SIZE * HIDDEN_SIZE;
    int size_ho = HIDDEN_SIZE * OUTPUT_SIZE;
    int size_h = HIDDEN_SIZE;
    int size_o = OUTPUT_SIZE;
    float *v;

    // Update the hidden weights with momentum – think of it as inertia helping us go forward.
    v = net->hidden_weights_momentum;
    #pragma omp parallel for
    cblas_sscal(size_hw, MOMENTUM, v, 1);
    #pragma omp parallel for
    for (int i = 0; i < size_hw; i++) {
        v[i] -= learning_rate * dw_hidden[i];
    }
    #pragma omp parallel for
    cblas_saxpy(size_hw, 1.0f, v, 1, net->hidden_weights, 1);

    // Now update the hidden biases
    v = net->hidden_bias_momentum;
    #pragma omp parallel for
    cblas_sscal(size_h, MOMENTUM, v, 1);
    #pragma omp parallel for
    for (int i = 0; i < size_h; i++) {
        v[i] -= learning_rate * db_hidden[i];
    }
    #pragma omp parallel for
    cblas_saxpy(size_h, 1.0f, v, 1, net->hidden_bias, 1);

    // Update the output weights
    v = net->output_weights_momentum;
    #pragma omp parallel for
    cblas_sscal(size_ho, MOMENTUM, v, 1);
    #pragma omp parallel for
    for (int i = 0; i < size_ho; i++) {
        v[i] -= learning_rate * dw_output[i];
    }
    #pragma omp parallel for
    cblas_saxpy(size_ho, 1.0f, v, 1, net->output_weights, 1);

    // And finally update the output biases
    v = net->output_bias_momentum;
    #pragma omp parallel for
    cblas_sscal(size_o, MOMENTUM, v, 1);
    #pragma omp parallel for
    for (int i = 0; i < size_o; i++) {
        v[i] -= learning_rate * db_output[i];
    }
    #pragma omp parallel for
    cblas_saxpy(size_o, 1.0f, v, 1, net->output_bias, 1);
}

/////////////////////////////////////////////
// Set up our working space for training (like our little desk)
// and then later clean it up.
/////////////////////////////////////////////
void initialize_training_resources(TrainingResources *res) {
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

void free_training_resources(TrainingResources *res) {
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

/////////////////////////////////////////////
// Load the MNIST training images and labels into memory.
// These are our raw digits!
/////////////////////////////////////////////
void load_mnist_data(unsigned char **train_images, unsigned char **train_labels) {
    *train_images = (unsigned char *)malloc(MNIST_TRAIN_SIZE * INPUT_SIZE);
    *train_labels = (unsigned char *)malloc(MNIST_TRAIN_SIZE);
    if (!*train_images || !*train_labels) {
        fprintf(stderr, "Bummer, not enough memory for training data!\n");
        exit(1);
    }
    read_idx_file("train-images-idx3-ubyte.gz", *train_images, MNIST_TRAIN_SIZE * INPUT_SIZE);
    read_idx_file("train-labels-idx1-ubyte.gz", *train_labels, MNIST_TRAIN_SIZE);
}

/////////////////////////////////////////////
// Prepare a mini-batch from our dataset by normalizing the images
// and converting the labels to a one-hot format.
/////////////////////////////////////////////
void prepare_batch(const unsigned char *images, const unsigned char *labels, int start_idx, float *batch_X,
                   float *batch_y_onehot) {
    for (int i = 0; i < BATCH_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            batch_X[i * INPUT_SIZE + j] = images[(start_idx + i) * INPUT_SIZE + j] / 255.0f;
        }
        memset(&batch_y_onehot[i * OUTPUT_SIZE], 0, OUTPUT_SIZE * sizeof(float));
        batch_y_onehot[i * OUTPUT_SIZE + labels[start_idx + i]] = 1.0f;
    }
}

/////////////////////////////////////////////
// This is our training loop.
// For each epoch and each mini-batch, we do a forward pass, calculate how wrong we are,
// backpropagate the error, and update the weights. We also check if we're doing better.
/////////////////////////////////////////////
void train_network(Network *net, unsigned char *aug_images, unsigned char *aug_labels, int total_samples,
                   TrainingResources *res) {
    int num_batches = total_samples / BATCH_SIZE;
    float best_accuracy = 0.0f;
    int no_improve = 0;
    for (int epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        float learning_rate = BASE_LR * powf(LR_DECAY, epoch);
        float epoch_loss = 0.0f;
        float epoch_acc = 0.0f;
        shuffle_data(aug_images, aug_labels, total_samples);
        for (int batch = 0; batch < num_batches; batch++) {
            int start_idx = batch * BATCH_SIZE;
            prepare_batch(aug_images, aug_labels, start_idx, res->batch_X, res->batch_y_onehot);
            forward_pass(net, res->batch_X, res->hidden_layer, res->output_layer);
            float batch_loss, batch_acc;
            compute_loss_accuracy(res->output_layer, res->batch_y_onehot, aug_labels, start_idx, &batch_loss, &batch_acc);
            epoch_loss += batch_loss;
            epoch_acc += batch_acc;
            backward_pass(net, res->batch_X, res->hidden_layer, res->output_layer, res->batch_y_onehot,
                          res->hidden_error, res->output_error, res->dw_hidden, res->dw_output, res->db_hidden, res->db_output);
            update_network(net, res->dw_hidden, res->dw_output, res->db_hidden, res->db_output, learning_rate);
            if (batch % PRINT_INTERVAL == 0) {
                printf("Epoch %d, Batch %d/%d, Loss: %.4f, Accuracy: %.2f%%\n", epoch+1, batch, num_batches, batch_loss, batch_acc * 100.0f);
            }
        }
        epoch_loss /= num_batches;
        epoch_acc /= num_batches;
        printf("Epoch %d/%d, Loss: %.4f, Accuracy: %.2f%%\n", epoch+1, NUM_EPOCHS, epoch_loss, epoch_acc * 100.0f);
        if (epoch_acc > best_accuracy) {
            best_accuracy = epoch_acc;
            no_improve = 0;
            printf("Great job! Saving these awesome weights...\n");
            save_weights(net);
        } else {
            no_improve++;
            if (no_improve >= PATIENCE) {
                printf("Seems like we're stuck. Early stopping. Best accuracy: %.2f%%\n", best_accuracy * 100.0f);
                break;
            }
        }
    }
}

/////////////////////////////////////////////
// Time to clean up all our network memory.
// We don’t want any leaks – our program is tidy!
/////////////////////////////////////////////
void free_network(Network *net) {
    free(net->hidden_weights);
    free(net->hidden_bias);
    free(net->output_weights);
    free(net->output_bias);
    free(net->hidden_weights_momentum);
    free(net->hidden_bias_momentum);
    free(net->output_weights_momentum);
    free(net->output_bias_momentum);
}

/////////////////////////////////////////////
// Save the network weights to a header file so we can use them later.
// This is our way of “printing” out what the network learned.
/////////////////////////////////////////////
void save_weights(Network *net) {
    FILE *f = fopen("src/weights.h", "w");
    if (!f) {
        fprintf(stderr, "Oops! Can't open weights.h for writing.\n");
        return;
    }
    fprintf(f, "#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n");
    fprintf(f, "#define INPUT_SIZE %d\n", INPUT_SIZE);
    fprintf(f, "#define HIDDEN_SIZE %d\n", HIDDEN_SIZE);
    fprintf(f, "#define OUTPUT_SIZE %d\n\n", OUTPUT_SIZE);
    fprintf(f, "static const float HIDDEN_WEIGHTS[INPUT_SIZE * HIDDEN_SIZE] = {\n");
    for (int i = 0; i < INPUT_SIZE * HIDDEN_SIZE; i++) {
        fprintf(f, "    %10.6ff%s", net->hidden_weights[i], (i + 1 < INPUT_SIZE * HIDDEN_SIZE) ? "," : "");
        if ((i + 1) % 8 == 0) fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
    fprintf(f, "static const float HIDDEN_BIAS[HIDDEN_SIZE] = {\n");
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        fprintf(f, "    %10.6ff%s", net->hidden_bias[i], (i + 1 < HIDDEN_SIZE) ? "," : "");
        if ((i + 1) % 8 == 0) fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
    fprintf(f, "static const float OUTPUT_WEIGHTS[HIDDEN_SIZE * OUTPUT_SIZE] = {\n");
    for (int i = 0; i < HIDDEN_SIZE * OUTPUT_SIZE; i++) {
        fprintf(f, "    %10.6ff%s", net->output_weights[i], (i + 1 < HIDDEN_SIZE * OUTPUT_SIZE) ? "," : "");
        if ((i + 1) % 8 == 0) fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
    fprintf(f, "static const float OUTPUT_BIAS[OUTPUT_SIZE] = {\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        fprintf(f, "    %10.6ff%s", net->output_bias[i], (i + 1 < OUTPUT_SIZE) ? "," : "");
        if ((i + 1) % 8 == 0) fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
    fprintf(f, "#endif /* WEIGHTS_H */\n");
    fclose(f);
    printf("Yay! We saved the weights to src/weights.h\n");
}

/////////////////////////////////////////////
// This is the heart of our program.
// It sets up the network, loads and augments the data, trains the network,
// and then cleans everything up. Simple as that.
/////////////////////////////////////////////
int main(void) {
    srand(RAND_SEED);
    Network net;
    initialize_network(&net);
    unsigned char *train_images, *train_labels;
    load_mnist_data(&train_images, &train_labels);
    unsigned char *aug_images = (unsigned char *)malloc(TOTAL_SAMPLES * INPUT_SIZE);
    unsigned char *aug_labels = (unsigned char *)malloc(TOTAL_SAMPLES);
    if (!aug_images || !aug_labels) {
        fprintf(stderr, "Oh no! Failed to get memory for augmented dataset.\n");
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
