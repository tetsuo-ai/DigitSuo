#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "neural_net.h"
#include "weights.h"
#include "draw_interface.h"
#include "utils.h"

static float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

NeuralNet *init_neural_net(void)
{
    FILE *debug_log = fopen(DEBUG_LOG_PATH, DEBUG_LOG_MODE);
    if (!debug_log)
    {
        fprintf(stderr, "Failed to open debug log in init_neural_net\n");
        return NULL;
    }

    fprintf(debug_log, "Initializing neural network\n");
    fflush(debug_log);

    NeuralNet *net = (NeuralNet *)malloc(sizeof(NeuralNet));
    if (!net)
    {
        fprintf(debug_log, "Failed to allocate neural network structure\n");
        fflush(debug_log);
        fclose(debug_log);
        return NULL;
    }

    net->hidden_weights = (float *)malloc(INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    net->hidden_bias = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    net->output_weights = (float *)malloc(HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    net->output_bias = (float *)malloc(OUTPUT_SIZE * sizeof(float));

    if (!net->hidden_weights || !net->hidden_bias || !net->output_weights || !net->output_bias)
    {
        fprintf(debug_log, "Failed to allocate weights/biases\n");
        fflush(debug_log);
        free_neural_net(net);
        fclose(debug_log);
        return NULL;
    }

    memcpy(net->hidden_weights, HIDDEN_WEIGHTS, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    memcpy(net->hidden_bias, HIDDEN_BIAS, HIDDEN_SIZE * sizeof(float));
    memcpy(net->output_weights, OUTPUT_WEIGHTS, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    memcpy(net->output_bias, OUTPUT_BIAS, OUTPUT_SIZE * sizeof(float));

    fprintf(debug_log, "Neural network initialized successfully\n");
    fflush(debug_log);
    fclose(debug_log);
    return net;
}

void free_neural_net(NeuralNet *net)
{
    if (net)
    {
        free(net->hidden_weights);
        free(net->hidden_bias);
        free(net->output_weights);
        free(net->output_bias);
        free(net);
    }
}

float *forward_pass(NeuralNet *net, float *input)
{
    FILE *debug_log = fopen("debug.log", "a");
    if (!debug_log)
        return NULL;

    fprintf(debug_log, "\n=== Starting Forward Pass ===\n");

    float *hidden = (float *)malloc(HIDDEN_SIZE * sizeof(float));
    float *output = (float *)malloc(OUTPUT_SIZE * sizeof(float));
    if (!hidden || !output)
    {
        fprintf(debug_log, "Memory allocation failed\n");
        free(hidden);
        free(output);
        fclose(debug_log);
        return NULL;
    }

    fprintf(debug_log, "Computing hidden layer with ReLU activation\n");
    for (int i = 0; i < HIDDEN_SIZE; i++)
    {
        float sum = net->hidden_bias[i];
        for (int j = 0; j < INPUT_SIZE; j++)
        {
            sum += input[j] * net->hidden_weights[j * HIDDEN_SIZE + i];
        }
        hidden[i] = relu(sum);
    }

    fprintf(debug_log, "Computing output layer with softmax activation\n");
    float max_val = -INFINITY;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        float sum = net->output_bias[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
        {
            sum += hidden[j] * net->output_weights[j * OUTPUT_SIZE + i];
        }
        output[i] = sum;
        if (sum > max_val)
            max_val = sum;
    }

    float sum = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        output[i] = expf(output[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        output[i] /= sum;
    }

    fprintf(debug_log, "\nPrediction probabilities:\n");
    for (int i = 0; i < OUTPUT_SIZE; i++)
    {
        fprintf(debug_log, "  %d: %.3f%%\n", i, output[i] * 100.0f);
    }

    fprintf(debug_log, "=== Forward Pass Complete ===\n\n");
    fflush(debug_log);
    fclose(debug_log);
    free(hidden);
    return output;
}

int get_prediction(float *output)
{
    int best_idx = 0;
    float best_conf = output[0];

    for (int i = 1; i < OUTPUT_SIZE; i++)
    {
        if (output[i] > best_conf)
        {
            best_conf = output[i];
            best_idx = i;
        }
    }

    return best_idx;
}
