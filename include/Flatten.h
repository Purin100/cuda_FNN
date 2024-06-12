/*
This file difines Flatten layer.
*/
#ifndef __FLATTEN_H__
#define __FLATTEN_H__

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <vector>
#include "Matrix.cuh"

using std::vector;

class Flatten
{
public:
    Flatten();
    ~Flatten();

    bool BuildLayer(const int input_row, const int input_col);

    void Forward(MYTYPE** _input, int row, int col);
    void Forward(Vector& _input);

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
