/*
This file difines Flatten layer.
*/
#ifndef __FLATTEN_H__
#define __FLATTEN_H__

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "Matrix.cuh"


class Flatten
{
public:
    Flatten();
    ~Flatten();

    bool BuildLayer(const int input_row, const int input_col);

    void Forward(MYTYPE** _input, int row, int col);

    //Now the flatten layer is the first layer of the neural network
    //So this function is useless for now
    void Backward(Vector& _loss);

    void DisplayOutput();
    void Save(const char* dir_name, const char* mode);
    Vector& GetOutput() {return output;}
    int GetSize() { return output.size();}

    Vector loss;
private:
    
    Vector output;
    int size;
};

#endif