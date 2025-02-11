#ifndef UTILS_H
#define UTILS_H

#include "draw_interface.h"

#define PREPROCESSING_TARGET_SIZE 18.0f
#define CONTRAST_THRESHOLD 0.3f
#define NORMALIZED_MAX_VALUE 255.0f
#define BINARY_THRESHOLD 128.0f

#define DEBUG_FILLED_CHAR '#'
#define DEBUG_EMPTY_CHAR '.'

#define DEBUG_LOG_PATH "debug.log"
#define DEBUG_LOG_MODE "a"

float *preprocess_grid(DrawGrid *grid);

#endif // UTILS_H