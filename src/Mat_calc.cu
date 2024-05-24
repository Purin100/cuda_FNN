#include "Mat_calc.cuh"

//c=a+b
__global__ void addVector(MYTYPE* c, MYTYPE* a, MYTYPE* b, const int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
        c[i] = a[i] + b[i];
}

//c=a-b
__global__ void subVector(MYTYPE* c, MYTYPE* a, MYTYPE* b, const int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
        c[i] = a[i] - b[i];
}

//element-wise multiplication for vector
//result will be saved in vec_a
__global__ void eleWiseMultV(MYTYPE* vec_a, MYTYPE* vec_b,const int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
        vec_a[i] *= vec_b[i];
}

//element-wise division for vector
//result will be saved in vec_a
__global__ void eleWiseDivV(MYTYPE* vec_a, MYTYPE* vec_b, const int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
        vec_a[i] /= vec_b[i];
}

//vector mutiplies a number
__global__ void numMultV(MYTYPE num, MYTYPE* vec, const int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
        vec[i] *= num;
}

__global__ void VecDevideNum(MYTYPE* vec, MYTYPE num, const int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        vec[i] /= num;
}

__global__ void VecMinusNum(MYTYPE* vec, const MYTYPE num, const int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        vec[i] -= num;
}

__global__ void VecPlusNum(MYTYPE* vec, const MYTYPE num, const int size)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size)
        vec[i] += num;
}

__global__ void Vecsqrt(MYTYPE* vec, const int size)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size)
        sqrt(vec[i]);
}

__global__ void MatMultNum(MYTYPE* mat, MYTYPE num, const int row, const int col)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < row && j < col)
        mat[i * col + j] *= num;
}

__global__ void MatSubMat(MYTYPE* mat_a, MYTYPE* mat_b, const int row, const int col)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < row && j < col)
        mat_a[i * col + j] -= mat_b[i * col + j];
}

__global__ void MatEledot(MYTYPE* mat_a, MYTYPE* mat_b, const int row, const int col)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < row && j < col)
        mat_a[i * col + j] *= mat_b[i * col + j];
}

__global__ void MatAddMat(MYTYPE* mat_a, MYTYPE* mat_b, const int row, const int col)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < row && j < col)
        mat_a[i * col + j] += mat_b[i * col + j];
}

mat_calc::mat_calc()
{
    _init();
}

mat_calc::~mat_calc()
{
    
}

void mat_calc::_init()
{
    cublasStatus_t e = cublasCreate(&handle);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
        printf("%d\n", e);
    }
}

void mat_calc::gemm_gpu(MYTYPE* mat_a, int row_a, int col_a, 
    MYTYPE* mat_b, int row_b, int col_b, MYTYPE* mat_c, int row_c,
    cublasOperation_t transpose_a, cublasOperation_t transpose_b)
{
    
    cublasStatus_t e;
    double alpha = 1.0, beta = 0.0;
    if (transpose_a == CUBLAS_OP_T && transpose_b == CUBLAS_OP_T)
    {
        assert(col_a == row_b);
        e = cublasDgemm(handle, transpose_a, transpose_b, row_a, col_b, col_a, &alpha, mat_a, col_a, mat_b, col_b, &beta, mat_c, row_c);
    }
    if (transpose_a == CUBLAS_OP_N && transpose_b == CUBLAS_OP_N)
    {
        assert(col_a == row_b);
        //e = cublasDgemm(handle, transpose_a, transpose_b, row_a, col_b, col_a, &alpha, mat_a, col_a, mat_b, row_b, &beta, mat_c, row_c);
        e = cublasDgemm(handle, transpose_a, transpose_b, col_b, row_a, col_a, &alpha, mat_b, col_b, mat_a, col_a, &beta, mat_c, col_b);
    }
    if (e != CUBLAS_STATUS_SUCCESS)
    {
        printf("%d\n", e);
        getchar();
    }
    cudaDeviceSynchronize();
}

void mat_calc::gemm_gpu_batched(MYTYPE* mat_a, int row_a, int col_a, 
    MYTYPE* mat_b, int row_b, int col_b, MYTYPE* mat_c, int row_c, 
    int batch_size, 
    cublasOperation_t transpose_a, cublasOperation_t transpose_b)
{
    cublasStatus_t e;
    double alpha = 1.0, beta = 0.0;
    cudaDeviceSynchronize();
}

void mat_calc::gemv_gpu(MYTYPE* mat_a, int row_a, int col_a, MYTYPE* vec, MYTYPE* res)
{
    cublasStatus_t e;
    double alpha = 1.0, beta = 0.0;
    e = cublasDgemv(handle, CUBLAS_OP_T, col_a, row_a, &alpha, mat_a, col_a, vec, 1, &beta, res, 1);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
        printf("error happens in gemv_gpu. error code: %d\n", e);
    }
    cudaDeviceSynchronize();
}

void mat_calc::gemv_gpu_batched(MYTYPE* mat_a, int row_a, int col_a, MYTYPE* vec, MYTYPE* res, int batch_size)
{
    cublasStatus_t e;
    double alpha = 1.0, beta = 0.0;
    //e = cublasDgemvBatched(handle, CUBLAS_OP_T, col_a, row_a, &alpha, mat_a, col_a, res, 1, &beta, res, 1, batch_size);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
        printf("error happens in gemv_gpu. error code: %d\n", e);
    }
    cudaDeviceSynchronize();
}

void mat_calc::VectorAdd(MYTYPE* c, MYTYPE* a, MYTYPE* b, int size)
{
    int threads = 32;
    int blocks = (size + threads - 1) / threads;
    addVector << <blocks, threads >> > (c, a, b, size);
    cudaDeviceSynchronize();
}

void mat_calc::VecMultNum(MYTYPE* vec, MYTYPE num, int size)
{
    cublasStatus_t e;
    e = cublasDscal(handle, size, &num, vec, 1);
    if (e != CUBLAS_STATUS_SUCCESS)
    {
        printf("Error happens in mat_calc::VecMultNum, cublasStatus error code: %d\n", e);
        getchar();
    }
    cudaDeviceSynchronize();
}

void mat_calc::VecDivNum(MYTYPE* vec, MYTYPE num, int size)
{
    int threads = 32;
    int blocks = (size + threads - 1) / threads;
    VecDevideNum << <blocks, threads >> > (vec, num, size);
    cudaDeviceSynchronize();
}

void mat_calc::VecSubNum(MYTYPE* vec, const MYTYPE num, const int size)
{
    int threads = 32;
    int blocks = (size + threads - 1) / threads;
    VecMinusNum <<<blocks, threads >>> (vec, num, size);
    cudaDeviceSynchronize();
}

void mat_calc::VecAddNum(MYTYPE* vec, const MYTYPE num, const int size)
{
    int threads = 32;
    int blocks = (size + threads - 1) / threads;
    VecPlusNum <<<blocks, threads >>> (vec, num, size);
    cudaDeviceSynchronize();
}

void mat_calc::VecEleMult(MYTYPE* vec_a, MYTYPE* vec_b, int size)
{
    int threads = 32;
    int blocks = (size + threads - 1) / threads;
    eleWiseMultV << <blocks, threads >> > (vec_a, vec_b, size);
    cudaDeviceSynchronize();
}

void mat_calc::VecEleDiv(MYTYPE* vec_a, MYTYPE* vec_b, int size)
{
    int threads = 32;
    int blocks = (size + threads - 1) / threads;
    eleWiseDivV <<<blocks,threads>>> (vec_a, vec_b, size);
    cudaDeviceSynchronize();
}

void mat_calc::VecSub(MYTYPE* vec_a, MYTYPE* vec_b, MYTYPE* vec_res, int size)
{
    int threads = 32;
    int blocks = (size + threads - 1) / threads;
    subVector<<<blocks,threads>>>(vec_res, vec_a, vec_b, size);
    cudaDeviceSynchronize();
}

void mat_calc::VecSqrt(MYTYPE* vec, int size)
{
    int threads = 32;
    int blocks = (size + threads - 1) / threads;
    Vecsqrt <<<blocks, threads >>> (vec, size);
    cudaDeviceSynchronize();
}

MYTYPE mat_calc::dot(MYTYPE* vec_a, MYTYPE* vec_b, const int size)
{
    MYTYPE res;
    assert(size > 0);
    cublasDdot(handle, size, vec_a, 1, vec_b, 1, &res);
    cudaDeviceSynchronize();
    return res;
}

MYTYPE mat_calc::VecAngle(MYTYPE* vec_a, MYTYPE* vec_b, const int size)
{
    MYTYPE res, dot_res;
    MYTYPE L2_a, L2_b;

    cublasDnrm2(handle, size, vec_a, 1, &L2_a);
    cudaDeviceSynchronize();
    cublasDnrm2(handle, size, vec_b, 1, &L2_b);
    cudaDeviceSynchronize();

    dot_res = dot(vec_a, vec_b, size);
    res = dot_res / (L2_a * L2_b);
    return res;
}

void mat_calc::VecNormalize(MYTYPE* vec, const int size)
{
    MYTYPE temp;
    assert(size > 0);
    assert(vec);

    // calculate Euclidean norm of the vector
    cublasDnrm2(handle, size, vec, 1, &temp);
    cudaDeviceSynchronize();

    int threads = 32;
    int blocks = (size + threads - 1) / threads;
    VecDevideNum <<<blocks,threads>>> (vec, temp, size);
    cudaDeviceSynchronize();
}

void mat_calc::MatrixMultNumber(MYTYPE* mat, MYTYPE number, int row, int col)
{
    cublasDscal(handle, row*col, &number, mat, 1);
    cudaDeviceSynchronize();
}


void mat_calc::MatrixDivNumber(MYTYPE* mat, MYTYPE num, int row, int col)
{
    cublasDscal(handle, row * col, &num, mat, 1);
    cudaDeviceSynchronize();
}

void mat_calc::MatrixSub(MYTYPE* mat_a, MYTYPE* mat_b, int row, int col)
{
    dim3 threads(16, 16, 1);
    dim3 blocks((row + threads.x - 1) / threads.x, (col + threads.y - 1) / threads.y, 1);
    MatSubMat << <blocks, threads >> > (mat_a, mat_b, row, col);
    cudaDeviceSynchronize();
}

__global__ void MatplusNum(MYTYPE* mat_a, int row, int col, MYTYPE num)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < row && j < col)
        mat_a[i * col + j] += num;
}
void mat_calc::MatrixAddNum(MYTYPE* mat_a, int row, int col, MYTYPE num)
{
    dim3 threads(16, 16, 1);
    dim3 blocks((row + threads.x - 1) / threads.x, (col + threads.y - 1) / threads.y, 1);
    MatplusNum <<<blocks, threads >>> (mat_a, row, col, num);
    cudaDeviceSynchronize();
}

void mat_calc::MatrixEleMult(MYTYPE* mat_a, MYTYPE* mat_b, int row, int col)
{
    dim3 threads(16, 16, 1);
    dim3 blocks((row + threads.x - 1) / threads.x, (col + threads.y - 1) / threads.y, 1);
    MatEledot<<<blocks,threads>>>(mat_a, mat_b, row, col);
    cudaDeviceSynchronize();
}

__global__ void MatEleDiv(MYTYPE* mat_a, MYTYPE* mat_b, int row, int col)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < row && j < col)
    {
        mat_a[i * col + j] /= mat_b[i * col + j];
    }
}
void mat_calc::MatrixEleDiv(MYTYPE* mat_a, MYTYPE* mat_b, int row, int col)
{
    dim3 threads(16, 16, 1);
    dim3 blocks((row + threads.x - 1) / threads.x, (col + threads.y - 1) / threads.y, 1);
    MatEleDiv <<<blocks, threads>>> (mat_a, mat_b, row, col);
    cudaDeviceSynchronize();
}

void mat_calc::MatrixTranspose(MYTYPE* src, MYTYPE* dst, const int row_src, const int col_src)
{
    MYTYPE alpha = 1.0, beta = 0.0;
    cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, row_src, col_src, &alpha, src, col_src, &beta, nullptr, row_src, dst, row_src);
}

__global__ void Matsqrt(MYTYPE* mat, const int row, const int col)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i < row && j < col)
        if (mat[i * col + j] >= 0.0)
            mat[i * col + j] = sqrt(mat[i * col + j]);
        else
            mat[i * col + j] = 0.0;
}
void mat_calc::MatSqrt(MYTYPE* mat, const int row, const int col)
{
    dim3 threads(16, 16, 1);
    dim3 blocks((row + threads.x - 1) / threads.x, (col + threads.y - 1) / threads.y, 1);
    Matsqrt <<<blocks, threads>>> (mat, row, col);
    cudaDeviceSynchronize();
}

void mat_calc::MatrixAdd(MYTYPE* mat_a, MYTYPE* mat_b, int row, int col)
{
    dim3 threads(16, 16, 1);
    dim3 blocks((row + threads.x - 1) / threads.x, (col + threads.y - 1) / threads.y, 1);
    MatAddMat <<<blocks, threads >>> (mat_a, mat_b, row, col);
    cudaDeviceSynchronize();
}