#pragma once
#include "utils.h"
#include "Matrix.cuh"
#include <opencv.hpp>
#include <string>

class PNGReader
{
public:
    PNGReader();
    PNGReader(_In_ cv::Mat& pic);
    ~PNGReader();

    bool ReadFile(_In_ const char* _filename);
    bool OpenFileInDir(_In_ std::string _dir);

    void Shrink(ShrinkMode _mode);

    //char success;
    Vector pixel;
    int label;
private:
    int width, height;
};