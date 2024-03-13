#ifndef __TXTREADER_H__
#define __TXTREADER_H__

#include <stdio.h>
#include <string.h>
#include <fstream>
#include <string>
#include "utils.cuh"

using std::string;

enum
{
    ZERO_TO_ONE = 0,
    MINUS_ONE_TO_ONE = 1
};

class TXTReader
{
public:
    TXTReader();
    ~TXTReader();

    bool ReadFile(const char* dataset_name, string& file_name);
    bool ReadCustomDataset(int width, int height, int channel, string& file_name);
    bool empty();
    bool isSame(int w, int h);

    int Width(){return this->width;};
    int Height(){return this->height;};

    void Shrink(char _mode = ZERO_TO_ONE);

    MYTYPE** Getdata(char which) const;
    const int GetChannel(){return channel;};
private:
    MYTYPE** data_c1;
    MYTYPE** data_c2;
    MYTYPE** data_c3;
    FILE *fp;

    int width, height;
    int channel;

};

#endif