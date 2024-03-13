#include "Flatten.h"

Flatten::Flatten()
{
    size = 0;
}

Flatten::~Flatten()
{

}

bool Flatten::BuildLayer(const int input_row, const int input_col)
{
    size = input_row * input_col;
    output = Vector(size);
    loss = Vector(size);
    return true;
}

void Flatten::Forward(MYTYPE** _input, int row, int col)
{
    int count = 0;
    if(!_input)
    {
        printf("ERROR: _input is null pointer.\n");
        getchar();
        std::abort();
        return;
    }
    if(row <= 0 || col <= 0)
    {
        printf("ERROR: invalid paraments. Please check values in paraments row and col, ensure row > 0 and col > 0.\n");
        getchar();
        std::abort();
        return;
    }
    size = row * col;
    /*if (output.empty())
    {
        output = Vector(size);
    }*/
    if (output.size() != size)
    {
        output.Realloc(size);
    }

   /* if (loss.empty())
        loss = Vector(size);*/
    if (loss.size() != size)
        loss.Realloc(size);

    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
        {
            output[count] = _input[i][j];
            count++;
        }
    output.DataTransfer(HostToDevice);
}

void Flatten::Backward(Vector& _loss)
{
    if (_loss.empty())
    {
        printf("ERROR: vector _loss in Flatten::Backward is empty.\n");
        getchar();
        return;
    }
}

void Flatten::DisplayOutput()
{
    for(int i = 0; i < size; i++)
        printf("%f ", output[i]);
    printf("\n");
}

void Flatten::Save(const char* dir_name, const char* mode)
{
    FILE* fp;
    fp = fopen(dir_name, mode);
    if(!fp)
        return;
    for(int i = 0; i < size; i++)
    {
        fprintf(fp, "%f\n", output[i]);
    }
    fprintf(fp, "\n");
    fclose(fp);
}