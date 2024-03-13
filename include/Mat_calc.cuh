/*
This file defined mat_calc class for matrix and vector calculation using cublas library.
Please ensure all the pointers send to functions in this class are pointers on the device(graph card or GPU)

All the functions in this file are based on "double" data type, and may have problems if you send a float type varible.
Because I called cublasD... functions for calculation.
Use cublasS... to replace those CUDA functions if you want to use float type.
e.g. use cublasSgemm to replace cublasDgemm
*/
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <assert.h>
#include "utils.cuh"



class mat_calc
{
public:
    mat_calc();
    ~mat_calc();

    //Manually call this function to release handle before you call cudaDeviceReset().
    void Destory() { cublasDestroy(handle); }

    //Function for matrix * matrix. It is suitable for situation like vector (m * 1) * vector (1 * n) = matrix (m * n).
    //Using cublasDgemm() function
    //mat_c = mat_a * mat_b
    void gemm_gpu(MYTYPE* mat_a, int row_a, int col_a,
        MYTYPE* mat_b, int row_b, int col_b, MYTYPE* mat_c, int row_c,
        cublasOperation_t transpose_a = CUBLAS_OP_T, cublasOperation_t transpose_b = CUBLAS_OP_T);

    //Function for matrix (m * n) * vector (n * 1).
    //Using cublasDgemv() function
    //res = mat_a * vec
    void gemv_gpu(MYTYPE* mat_a, int row_a, int col_a, MYTYPE* vec, MYTYPE* res);

    //c = a + b
    void VectorAdd(MYTYPE* c, MYTYPE* a, MYTYPE* b, int size);

    void VecMultNum(MYTYPE* vec, MYTYPE num, int size);
    void VecDivNum(MYTYPE* vec, MYTYPE num, int size);

    //Assume there is a vector v=[v0,v1,v2], and a number b
    //this function allows v-b, which means [v0-b, v1-b, v2-b]
    void VecSubNum(MYTYPE* vec, const MYTYPE num, const int size);

    //Vector element-wise multiply
    //e.g., vec_a = (a0, a1), vec_b = (b0, b1), the result will be a vector with tow elements a0*b0 and a1*b1
    //vec_a = vec_a * vec_b
    void VecEleMult(MYTYPE* vec_a, MYTYPE* vec_b, int size);

    //vec_res = vec_a - vec_b
    void VecSub(MYTYPE* vec_a, MYTYPE* vec_b, MYTYPE* vec_res, int size);

    void MatrixMultNumber(MYTYPE* mat, MYTYPE number, int row, int col);

    //mat_a -= mat_b
    void MatrixSub(MYTYPE* mat_a, MYTYPE* mat_b, int row, int col);

    //mat_a += mat_b
    void MatrixAdd(MYTYPE* mat_a, MYTYPE* mat_b, int row, int col);

    //Matrix element-wise multiply
    //mat_a *= mat_b
    void MatrixEleMult(MYTYPE* mat_a, MYTYPE* mat_b, int row, int col);

    void Mat_minusClear(MYTYPE* mat, const int row, const int col);

private:
    //cublas library handle, using for call cuda matrix calculation library.
    //it will be created automaticly in the constructor
    cublasHandle_t handle;

    void _init();
};
