#pragma once
#include <stdio.h>
#include <random>
#include <algorithm>
#include <string>
#include "Activation.cuh"
#include "optimizer.h"
#include "Matrix.cuh"
#include "utils.cuh"
#include "Flatten.h"

using std::string;

class Dense
{
public:
    Dense() {};
    ~Dense();

    bool BuildLayer(int input_units, int output_units, 
        const char* activation, 
        const char* optimizer, 
        bool isfreeze = false);

    void Forward(Dense* pre_layer);
    void Forward(Vector& _input, const int element_num);

    //Functions for BP with different inputs
    void Backward(Vector& _loss, Dense* pre_layer, bool update);//This one is for the last layer in the neural network
    void Backward(Dense* pre_layer, bool update);
    void Backward(Flatten* pre_layer, bool update);
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

    void OptReset() { adam.SetpReset(); }

private:
    Matrix weight;
    Matrix weight_t;//transposed weight matrix
    Matrix grad_sample;//save gradient for each sample in a batch
    Matrix weight_grad;//matrix contains gradient values of weight matrix
    Matrix save_grad;
    Matrix grad_direction;

    MYTYPE lr = 0.001;

    Vector output;
    Vector input_frequence, output_frequence;
    Vector bias;
    Vector bias_batch;//save gradient in one batch
    Vector input;
    Vector local_out;

    //activation function and its gradient
    void (*activation)(MYTYPE* input, MYTYPE* output, const int size) = nullptr;
    void (*gradient)(MYTYPE* input, MYTYPE* output, const int size) = nullptr;
    MYTYPE* temp_grad = nullptr;
    int input_units = 0, output_units = 0;
    int sample_num = 0;
    bool freeze_weight = false;
    Adam adam;

    //std::uniform_real_distribution<MYTYPE> xavier;
};