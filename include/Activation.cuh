#pragma once
//#ifndef __CUDACC__
//#define __CUDACC__
//#endif
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include "utils.cuh"


__global__ void Tanh(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Relu(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Sigmoid(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Linear(MYTYPE* x, MYTYPE* output, const int num);
__global__ void LeakyRelu(MYTYPE* x, MYTYPE* output, const int num);

__global__ void Tanh_gradient(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Relu_gradient(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Linear_gradient(MYTYPE* x, MYTYPE* output, const int num);
__global__ void Sigmoid_gradient(MYTYPE* x, MYTYPE* output, const int num);
__global__ void LeakyRelu_gradient(MYTYPE* x, MYTYPE* output, const int num);

//__global__ void Softmax(MYTYPE* x, MYTYPE* output, const int num);
MYTYPE Cross_Entropy(MYTYPE* one_hot, MYTYPE* x, const int classnum, const int sample_num = 1);
MYTYPE MSE(MYTYPE* one_hot, MYTYPE* x, const int classnum);

static void Softmax(MYTYPE* x, MYTYPE* result, int classNum)
{
    MYTYPE sum = 0.0f, max=x[0];

    for (int i = 0; i < classNum; i++)
        if (x[i] > max)
            max = x[i];

    if (!x)
        return;
    if (!result)
        return;

    for (int i = 0; i < classNum; i++)
    {
        //x[i] -= max;
        if (isnan(x[i]))
        {
            printf("nan dected in softmax. %d\n", i);
            getchar();
        }
        sum += exp(x[i]);
    }

    for (int i = 0; i < classNum; i++)
        result[i] = exp(x[i]) / sum;

}