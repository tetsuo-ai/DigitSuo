#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "utils.h"

typedef struct
{
    int min_x;
    int max_x;
    int min_y;
    int max_y;
    int total_points;
} GridBounds;

typedef struct
{
    int width;
    int height;
    float center_x;
    float center_y;
} GridDimensions;

static FILE *open_debug_log(void)
{
    FILE *debug_log = fopen(DEBUG_LOG_PATH, DEBUG_LOG_MODE);
    if (!debug_log)
    {
        fprintf(stderr, "Failed to open debug log\n");
        return NULL;
    }
    fprintf(debug_log, "\n=== Starting Preprocessing ===\n");
    return debug_log;
}

static void close_debug_log(FILE *debug_log)
{
    if (debug_log)
    {
        fprintf(debug_log, "=== Preprocessing Complete ===\n\n");
        fflush(debug_log);
        fclose(debug_log);
    }
}

static GridBounds find_grid_bounds(const DrawGrid *grid)
{
    GridBounds bounds = {.min_x = GRID_SIZE, .max_x = 0, .min_y = GRID_SIZE, .max_y = 0, .total_points = 0};

    for (int y = 0; y < GRID_SIZE; y++)
    {
        for (int x = 0; x < GRID_SIZE; x++)
        {
            if (grid->cells[y][x])
            {
                bounds.min_x = fmin(bounds.min_x, x);
                bounds.max_x = fmax(bounds.max_x, x);
                bounds.min_y = fmin(bounds.min_y, y);
                bounds.max_y = fmax(bounds.max_y, y);
                bounds.total_points++;
            }
        }
    }

    return bounds;
}

static GridDimensions calculate_dimensions(const GridBounds *bounds)
{
    GridDimensions dims = {.width = bounds->max_x - bounds->min_x + 1,
                           .height = bounds->max_y - bounds->min_y + 1,
                           .center_x = (bounds->min_x + bounds->max_x) / 2.0f,
                           .center_y = (bounds->min_y + bounds->max_y) / 2.0f};
    return dims;
}

static float bilinear_interpolate(const DrawGrid *grid, float src_x, float src_y)
{
    int x0 = (int)src_x;
    int y0 = (int)src_y;
    float fx = src_x - x0;
    float fy = src_y - y0;

    if (x0 < 0 || x0 >= GRID_SIZE - 1 || y0 < 0 || y0 >= GRID_SIZE - 1)
    {
        return 0.0f;
    }

    float v00 = grid->cells[y0][x0];
    float v01 = grid->cells[y0][x0 + 1];
    float v10 = grid->cells[y0 + 1][x0];
    float v11 = grid->cells[y0 + 1][x0 + 1];

    return (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v01 + (1 - fx) * fy * v10 + fx * fy * v11;
}

static void debug_print_grid(FILE *debug_log, const float *input)
{
    fprintf(debug_log, "\nPreprocessed digit:\n");
    for (int y = 0; y < GRID_SIZE; y++)
    {
        for (int x = 0; x < GRID_SIZE; x++)
        {
            fprintf(debug_log, "%c",
                    input[y * GRID_SIZE + x] > BINARY_THRESHOLD ? DEBUG_FILLED_CHAR : DEBUG_EMPTY_CHAR);
        }
        fprintf(debug_log, "\n");
    }
}

float *preprocess_grid(DrawGrid *grid)
{
    FILE *debug_log = open_debug_log();
    if (!debug_log)
    {
        return NULL;
    }

    GridBounds bounds = find_grid_bounds(grid);
    if (bounds.total_points == 0)
    {
        fprintf(debug_log, "No content found in grid\n");
        close_debug_log(debug_log);
        return NULL;
    }

    GridDimensions dims = calculate_dimensions(&bounds);
    fprintf(debug_log, "Content bounds: (%d,%d) to (%d,%d)\n", bounds.min_x, bounds.min_y, bounds.max_x, bounds.max_y);
    fprintf(debug_log, "Dimensions: %dx%d\n", dims.width, dims.height);

    float scale = PREPROCESSING_TARGET_SIZE / fmax(dims.width, dims.height);
    fprintf(debug_log, "Scale factor: %.3f\n", scale);

    float *input = (float *)calloc(GRID_SIZE * GRID_SIZE, sizeof(float));
    if (!input)
    {
        fprintf(debug_log, "Failed to allocate memory for input\n");
        close_debug_log(debug_log);
        return NULL;
    }

    float target_center_x = GRID_SIZE / 2.0f;
    float target_center_y = GRID_SIZE / 2.0f;

    for (int y = 0; y < GRID_SIZE; y++)
    {
        for (int x = 0; x < GRID_SIZE; x++)
        {
            float src_x = ((x - target_center_x) / scale + dims.center_x);
            float src_y = ((y - target_center_y) / scale + dims.center_y);

            float value = bilinear_interpolate(grid, src_x, src_y);
            value = value > CONTRAST_THRESHOLD ? 1.0f : value;
            input[y * GRID_SIZE + x] = value * NORMALIZED_MAX_VALUE;
        }
    }

    debug_print_grid(debug_log, input);
    close_debug_log(debug_log);

    return input;
}