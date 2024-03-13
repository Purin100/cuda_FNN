/*
Matrix and vector class are defined in this file.
These classes are used as basic parts in all kinds of layers in this project.
*/

#pragma once
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "utils.cuh"
#include "Mat_calc.cuh"


class Vector;
class Matrix;
static mat_calc mc;//an object for matrix (and vector) calculation

//Generate an identity matrix sized in num * num.
Matrix Identity(const int num);

//Generate a matrix sized in num * num with all elements are zero
Matrix Zeros(const int num);

//Generate a matrix sized in row * col with all elements are one
Matrix Ones(const int row, const int col);

//In CUDA, a matrix is stored in column-mayjor. While in some CPU, a matrix is stroed in row-mayjor (like C++)
//If we need to copy matrix from GPU to CPU, and do something on CPU matrix, we need to change the matrix into row-mayjor.
//ATTENTION: src and dst should be two different pointers.
void Colmaj2Rowmaj(MYTYPE* src, MYTYPE* dst, int src_row, int src_col);

#define HostToDevice 1
#define DeviceToHost 2

class Vector
{
public:
    Vector() {};
    ~Vector() { if (data) delete[] data; cudaFree(dev_data); };
    Vector(const int element_num);
    Vector(const Vector& v);
    Vector(MYTYPE* data, const int element_num);//Initialize a vector using an exsisting one dim array on CPU.

    MYTYPE* GetVec() { return data; }
    MYTYPE* GetDevVec() { return dev_data; }

    //copy a vector from CPU to GPU (HostToDevice), or from GPU to CPU (DeviceToHost)
    void DataTransfer(int trans_label);

    //free memory occupied by this vector (on CPU and on GPU) and reallocate a new space for this vector
    //all content in the vector will be cleaned
    //this function will do nothing if element_num == this->element_num
    void Realloc(int element_num);

    void showVector() { printf("Vector:"); for (int i = 0; i < element_num; i++) printf("%f ", data[i]); printf("\n"); }
    bool empty();
    int size() { return element_num; }

    MYTYPE& operator[](int i);
    Vector& operator+(const Vector& v);
    Vector& operator=(const Vector& v);
    //friend Vector operator-(const Vector& vec_a, const Vector& vec_b);

    Vector& operator-=(const Vector& v);
    Vector& operator-=(const MYTYPE num);
    Vector& operator+=(const Vector& v);
    Vector& operator*=(const MYTYPE num);
    Vector& operator/=(const MYTYPE num);

    friend Vector operator*(const MYTYPE num, const Vector& v);
    friend Vector operator*(const Vector& v, const MYTYPE num);
    friend Vector operator-(const Vector& v, const MYTYPE num);
    friend Vector operator-(const Vector& vec_a, const Vector& vec_b);
    
private:
    MYTYPE* data = nullptr;//data on CPU
    MYTYPE* dev_data = nullptr;//data on GPU
    int element_num = 0;//how many numbers in this vector
protected:
    void _init(const int element_num);
};

//Vector operator*(const MYTYPE num, const Vector& v);
//Vector operator*(const Vector& v, const MYTYPE num);

class Matrix
{
public:
    Matrix() {};
    Matrix(int row, int col);
    Matrix(const Matrix& m);
    ~Matrix();

    inline int rows() { return row; }
    inline int cols() { return col; }
    inline int size() { return element_num; }
    MYTYPE* GetMat() { return data; }
    MYTYPE* GetDevMat() { return dev_data; }
    Vector RowSlice(const int which_row);//Get a row in the matrix (in Vector type)

    //Fill a row in the matrix with given vector vec if which >= 0 && which < this->row.
    //If which < 0 or which > this->row, then fill all rows in the matrix with given vector vec.
    void RowFill(Vector vec, const int which = -1);

    void Zeroreset();//set all elements in the matrix to zero

    //copy a matrix from CPU to GPU (HostToDevice), or from GPU to CPU (DeviceToHost)
    void DataTransfer(int trans_label);
    void showMat();//print data on CPU

    bool empty();

    MYTYPE& operator()(int i, int j);//you can gain access to an element at i-th row and j-th column
    Matrix& operator=(const Matrix& m);

    friend Matrix operator-(const Matrix& m, const Matrix& n);
    friend Matrix operator+(const Matrix& m, const Matrix& n);

    Matrix& operator-=(const Matrix& m);
    Matrix& operator+=(const Matrix& m);
    Matrix& operator*=(const MYTYPE num);
private:
    //In CUDA, matrix is stored in an one-dim array,
    //so I use an one-dim pointer
    MYTYPE* data = nullptr;//data on CPU
    MYTYPE* dev_data = nullptr;//data on GPU
    int row = 0, col = 0;
    int element_num = 0;
protected:
    void _init(const int row, const int col);
};