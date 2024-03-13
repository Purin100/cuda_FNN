#pragma once
#include <stdio.h>
#include <random>
#include <algorithm>
#include <string>
#include "Activation.cuh"
#include "Matrix.cuh"
#include "utils.cuh"
#include "Flatten.h"
#pragma comment(lib,"curand.lib")

using std::string;

class Dense
{
public:
    Dense() {};
    ~Dense();

    bool BuildLayer(int input_units, int output_units, int node_num, const char* activation);

    void Forward(Dense* pre_layer);
    void Forward(Vector& _input, const int element_num);

    //Functions for BP with different inputs
    void Backward(Vector& _loss, Dense* pre_layer, const int num);//This one is for the last layer in the neural network

    void Backward(Dense* pre_layer);
    void Backward(Flatten* pre_layer);
    void Backward(Vector& _input);

    void Setlr(MYTYPE _lr) { 
        if(_lr>0.0) lr = _lr;
    }
    void lrDecay(const int now_epoch);//function that decreases the learning rate
    Vector& Getoutput() { return output; }

    void Save(string& _dir, int which);

    void SetWeight(Matrix _w)
    {
        if (!_w.empty())
            weight = _w;
    }

    Vector loss;//vector contains the loss values in this layer
private:
    int input_units = 0, output_units = 0;
    
    Matrix weight;
    Matrix weight_t;//transposed weight matrix
    Matrix weight_grad;//matrix contains gradient values of weight matrix
    Matrix split;

    MYTYPE lr = 0.001;

    Vector output;
    Vector local_out;
    Vector bias;
    

    //activation function and its gradient
    void (*activation)(MYTYPE* input, MYTYPE* output, const int size) = nullptr;
    void (*gradient)(MYTYPE* input, MYTYPE* output, const int size) = nullptr;
    MYTYPE* temp_grad = nullptr;

    std::uniform_real_distribution<MYTYPE> xavier;
};