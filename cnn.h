// File: cnn.h
#ifndef CNN_H
#define CNN_H

#include <Arduino.h>

#define DATA_ROWS   48
#define DATA_COLS   13
#define KERNEL_SIZE 3
#define POOL_SIZE   2

// dimensions after pooling
#define ROWS_P1 (DATA_ROWS / POOL_SIZE)    // 24
#define COLS_P1 (DATA_COLS / POOL_SIZE)    // 6
#define ROWS_P2 (ROWS_P1   / POOL_SIZE)    // 12
#define COLS_P2 (COLS_P1   / POOL_SIZE)    // 3
#define FLAT_SIZE (ROWS_P2 * COLS_P2)      // 36

#define N_HIDDEN   20
#define N_CLASSES  10

// --- trained parameters (replace the ... with Python-exported arrays) ---
extern const float kernel1[KERNEL_SIZE][KERNEL_SIZE];
extern const float kernel2[KERNEL_SIZE][KERNEL_SIZE];
extern const float weights_fc1[N_HIDDEN][FLAT_SIZE];
extern const float biases_fc1 [N_HIDDEN];
extern const float weights_fc2[N_CLASSES][N_HIDDEN];
extern const float biases_fc2 [N_CLASSES];
// -----------------------------------------------------------------------

// convolution + relu, pooling, flatten
void conv2d_same_relu_1(const float in[DATA_ROWS][DATA_COLS], float out[DATA_ROWS][DATA_COLS]);
void pool2d_1        (const float in[DATA_ROWS][DATA_COLS], float out[ROWS_P1][COLS_P1]);
void conv2d_same_relu_2(const float in[ROWS_P1][COLS_P1], float out[ROWS_P1][COLS_P1]);
void pool2d_2        (const float in[ROWS_P1][COLS_P1], float out[ROWS_P2][COLS_P2]);
void flatten2d       (const float in[ROWS_P2][COLS_P2],    float out[FLAT_SIZE]);
void cnn             (const float input[DATA_ROWS][DATA_COLS], float output[FLAT_SIZE]);

// dense + classification
void dense_relu     (const float in[FLAT_SIZE], float out[N_HIDDEN]);
void dense_logits   (const float in[N_HIDDEN],  float out[N_CLASSES]);
int  argmax         (const float arr[], int len);

#endif // CNN_H
