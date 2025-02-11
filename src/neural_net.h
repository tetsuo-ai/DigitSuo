#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "draw_interface.h"

typedef struct
{
    float *hidden_weights;
    float *hidden_bias;
    float *output_weights;
    float *output_bias;
} NeuralNet;

NeuralNet *init_neural_net(void);
void free_neural_net(NeuralNet *net);
float *forward_pass(NeuralNet *net, float *input);
int get_prediction(float *output);

#endif // NEURAL_NET_H
