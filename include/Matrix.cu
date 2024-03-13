#include "Matrix.cuh"
#include "Mat_calc.cuh"

//Vector part
Vector::Vector(const int element_num)
{
    _init(element_num);
}

Vector::Vector(const Vector& v)
{
    assert(v.element_num >= 0);
    {
        if (element_num == 0)
            _init(v.element_num);
        memcpy(this->data, v.data, sizeof(MYTYPE) * this->element_num);
        cudaMemcpy(dev_data, this->data, sizeof(MYTYPE) * this->element_num, cudaMemcpyHostToDevice);
    }
}

Vector::Vector(MYTYPE* data, const int element_num)
{
    if (element_num > 0 && data)
    {
        _init(element_num);
        memcpy(this->data, data, sizeof(MYTYPE) * this->element_num);
        cudaMemcpy(dev_data, data, sizeof(MYTYPE) * this->element_num, cudaMemcpyHostToDevice);
    }

}

void Vector::DataTransfer(int trans_label)
{
    if (trans_label == HostToDevice)
        cudaMemcpy(dev_data, data, sizeof(MYTYPE) * element_num, cudaMemcpyHostToDevice);
    if (trans_label == DeviceToHost)
        cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
}

void Vector::Realloc(int element_num)
{
    if (element_num == this->element_num)
        return;
    if (data)
    {
        delete[] data;
        data = nullptr;
    }
    cudaFree(dev_data);
    _init(element_num);
}

void Vector::_init(const int element_num)
{
    if (element_num > 0)
    {
        this->element_num = element_num;
        data = new MYTYPE[element_num];
        cudaMalloc((void**)&dev_data, sizeof(MYTYPE) * element_num);
    }
}

bool Vector::empty()
{
    return element_num == 0 || data == nullptr;
}

MYTYPE& Vector::operator[](int i)
{
    assert(data && i < this->element_num && i >= 0);
    return this->data[i];
}

Vector& Vector::operator+(const Vector& v)
{
    assert(v.element_num == this->element_num);
    mc.VectorAdd(this->dev_data, v.dev_data, this->dev_data, element_num);
    cudaMemcpy(data, dev_data, element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    return *this;
}

Vector& Vector::operator+=(const Vector& v)
{
    assert(v.element_num == this->element_num);
    mc.VectorAdd(this->dev_data, v.dev_data, this->dev_data, element_num);
    cudaMemcpy(data, dev_data, element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    return *this;
}

Vector& Vector::operator=(const Vector& v)
{
    assert(v.element_num > 0);
    if (this != &v)
    {
        if (this->element_num > 0)
        {
            assert(this->element_num == v.element_num);
            memcpy(this->data, v.data, sizeof(MYTYPE) * element_num);
        }
        else
        {
            this->element_num = v.element_num;
            this->_init(v.element_num);
            memcpy(this->data, v.data, sizeof(MYTYPE) * element_num);
        }
        cudaMemcpy(dev_data, v.dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToDevice);
    }
    return *this;
}

Vector operator-(const Vector& v, const MYTYPE num)
{
    Vector t = v;
    mc.VecSubNum(t.dev_data, num, t.element_num);
    return t;
}

Vector operator-(const Vector& vec_a, const Vector& vec_b)
{
    assert(vec_a.element_num == vec_b.element_num);
    Vector t = vec_a;
    mc.VecSub(t.dev_data, vec_b.dev_data, t.dev_data, t.element_num);
    cudaMemcpy(t.data, t.dev_data, t.element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    return t;
}

Vector& Vector::operator-=(const Vector& v)
{
    assert(this->element_num == v.element_num);
    mc.VecSub(dev_data, v.dev_data, dev_data, element_num);
    cudaMemcpy(data, dev_data, element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    return *this;
}

Vector& Vector::operator-=(const MYTYPE num)
{
    mc.VecSubNum(dev_data, num, element_num);
    cudaMemcpy(data, dev_data, element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    return *this;
}

Vector operator*(const Vector& v, const MYTYPE num)
{
    Vector t;
    t = v;
    assert(!t.empty());
    mc.VecMultNum(t.GetDevVec(), num, t.element_num);
    t.DataTransfer(DeviceToHost);
    return t;
}

Vector operator*(const MYTYPE num, const Vector& v)
{
    Vector t;
    t = v;
    assert(!t.empty());
    mc.VecMultNum(t.GetDevVec(), num, t.element_num);
    t.DataTransfer(DeviceToHost);
    return t;
}

Vector& Vector::operator*=(const MYTYPE num)
{
    if (this->element_num > 0)
    {
        mc.VecMultNum(this->dev_data, num, this->element_num);
        cudaMemcpy(data, dev_data, element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    }
    return *this;
}

Vector& Vector::operator/=(const MYTYPE num)
{
    assert(num != 0.0 && this->element_num > 0);
    mc.VecDivNum(dev_data, num, element_num);
    cudaMemcpy(data, dev_data, element_num * sizeof(MYTYPE), cudaMemcpyDeviceToHost);
    return *this;
}


//Matrix part
Matrix::Matrix(int row, int col)
{
    _init(row, col);
}

void Matrix::_init(int row, int col)
{
    assert(row > 0 && col > 0);
    this->row = row, this->col = col;
    element_num = row * col;
    data = new MYTYPE[element_num];
    cudaMalloc((void**)&dev_data, sizeof(MYTYPE) * element_num);
}

Matrix::Matrix(const Matrix& m)
{
    _init(m.row, m.col);
    memcpy(data, m.data, sizeof(MYTYPE) * element_num);
    cudaMemcpy(dev_data, m.dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToDevice);
}

Matrix::~Matrix()
{
    if (data)
    {
        delete[] data;
        data = nullptr;
    }
    cudaFree(dev_data);
}

MYTYPE& Matrix::operator()(int i, int j)
{
    assert(i >= 0 && i < row && j >= 0 && j < col && data);
    return data[i * col + j];
}

Matrix& Matrix::operator=(const Matrix& m)
{
    if (this != &m)
    {
        if (this->element_num != 0)
            assert(m.row == this->row && m.col == this->col);
        else
            this->_init(m.row, m.col);

        if(data)
            memcpy(data, m.data, sizeof(MYTYPE) * element_num);
        if(dev_data)
            cudaMemcpy(dev_data, m.dev_data, sizeof(MYTYPE)*element_num, cudaMemcpyDeviceToDevice);
    }
    return *this;
}

Matrix operator-(const Matrix& m, const Matrix& n)
{
    assert(n.col == m.col && n.row == m.row);
    Matrix t;
    t = m;
    mc.MatrixSub(t.dev_data, n.dev_data, m.row, m.col);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Matrix operator+(const Matrix& m, const Matrix& n)
{
    assert(n.col == m.col && n.row == m.row);
    Matrix t = m;
    mc.MatrixAdd(t.dev_data, n.dev_data, m.row, m.col);
    cudaMemcpy(t.data, t.dev_data, sizeof(MYTYPE) * t.element_num, cudaMemcpyDeviceToHost);
    return t;
}

Matrix& Matrix::operator-=(const Matrix& m)
{
    assert(this->col == m.col && this->row == m.row);
    mc.MatrixSub(this->dev_data, m.dev_data, row, col);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
    return *this;
}

Matrix& Matrix::operator+=(const Matrix& m)
{
    assert(this->row == m.row && this->col == m.col);
    mc.MatrixAdd(this->dev_data, m.dev_data, row, col);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
    return *this;
}

Matrix& Matrix::operator*=(const MYTYPE num)
{
    assert(this->element_num > 0);
    mc.MatrixMultNumber(dev_data, num, row, col);
    cudaMemcpy(data, dev_data, sizeof(MYTYPE) * row * col, cudaMemcpyDeviceToHost);
    return *this;
}

Vector Matrix::RowSlice(const int which_row)
{
    assert(which_row >= 0 && which_row < row);
    Vector v(col);
    for (int i = 0; i < col; i++)
        v[i] = data[which_row * col + i];
    return v;
}

void Matrix::DataTransfer(int trans_label)
{
    if (trans_label == HostToDevice)
        cudaMemcpy(dev_data, data, sizeof(MYTYPE) * element_num, cudaMemcpyHostToDevice);
    if (trans_label == DeviceToHost)
        cudaMemcpy(data, dev_data, sizeof(MYTYPE) * element_num, cudaMemcpyDeviceToHost);
}

void Matrix::showMat()
{
    for (int i = 0; i < row; i++)
    {
        for (int j = 0; j < col; j++)
            printf("%f ", data[i * col + j]);
        printf("\n");
    }
}

bool Matrix::empty()
{
    if (element_num <= 0 || !data || !dev_data)
        return true;
    else
        return false;
}

void Matrix::RowFill(Vector vec, const int which)
{
    assert(!vec.empty());
    assert(vec.size() == col);
    if (which >= 0 && which < row)
    {
        memcpy(&data[col * which], vec.GetVec(), sizeof(MYTYPE) * col);
    }
    else
    {
        for (int i = 0; i < element_num; i += col)
            memcpy(&data[i], vec.GetVec(), sizeof(MYTYPE) * col);
    }
    cudaMemcpy(dev_data, data, sizeof(MYTYPE) * element_num, cudaMemcpyHostToDevice);
}

void Matrix::Zeroreset()
{
    assert(data != nullptr && dev_data != nullptr);
    memset(data, 0, sizeof(MYTYPE) * element_num);
    cudaMemset(dev_data, 0, sizeof(MYTYPE) * element_num);
}

Matrix Identity(const int num)
{
    Matrix mat = Zeros(num);
    for (int i = 0; i < num; i++)
        mat(i, i) = 1.0;
    mat.DataTransfer(HostToDevice);
    return mat;
}
Matrix Zeros(const int num)
{
    Matrix mat(num, num);
    for (int i = 0; i < num; i++)
        for (int j = 0; j < num; j++)
            mat(i, j) = 0.0;
    mat.DataTransfer(HostToDevice);
    return mat;
}

Matrix Ones(const int row, const int col)
{
    Matrix mat(row, col);
    for (int i = 0; i < row; i++)
        for (int j = 0; j < col; j++)
            mat(i, j) = 1.0;
    mat.DataTransfer(HostToDevice);
    return mat;
}

void Colmaj2Rowmaj(MYTYPE* src, MYTYPE* dst, int src_row, int src_col)
{
    if (src == nullptr || dst == nullptr)
    {
        printf("ERROR: src or dst in function Colmaj2Rowmaj(MYTYPE*, MYTYPE*) is null pointer.\n");
        getchar();
        return;
    }

    int idx_src = 0, idx_dst = 0;
    for(; idx_src < src_row; idx_src++)
        for (int i = 0; i < src_col; i++)
        {
            dst[idx_dst] = src[idx_src + src_row * i];
            idx_dst++;
        }
}