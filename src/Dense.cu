#include "Dense.cuh"
#include <ctime>

static std::uniform_real_distribution<MYTYPE> lr_dis(0.0001, 0.001);
static std::random_device dense_rd;
static std::mt19937_64 dense_weight(dense_rd());

Dense::~Dense()
{
    if (temp_grad)
    {
        delete[] temp_grad;
        temp_grad = nullptr;
    }
}

bool Dense::BuildLayer(int input_units, int output_units, int node_num, const char* activation)
{
    if (activation == nullptr)
    {
        printf("ERROR: null pointer activation.\n");
        getchar();
        return false;
    }
    if (input_units < 0 || output_units <= 0)
    {
        printf("ERROR: invalid arguments. Please ensure input_units >= 0, and output_units > 0.");
        getchar();
        return false;
    }
    if (node_num <= 0)
    {
        printf("ERROR: invalid argument node_num. Please ensure node_num > 0.");
        getchar();
        return false;
    }

    this->input_units = input_units;
    this->output_units = output_units;

    weight = Matrix(output_units, input_units);
    weight_grad = Matrix(output_units, input_units);
    temp_grad = new MYTYPE[output_units * input_units]{ 0.0 };
    weight_t = Matrix(input_units, output_units);

    output = Vector(output_units);
    loss = Vector(output_units);
    bias = Vector(output_units);

    //Set activation function and relevant gradient function
    //Softmax DOES NOT have gradient function, and cannot be used in hidden layers
    if (mystrcmp(activation, "tanh") == 0)
    {
        this->activation = &Tanh;
        this->gradient = &Tanh_gradient;
        goto ENDACTIVATON;
    }
    if (mystrcmp(activation, "relu") == 0)
    {
        this->activation = &Relu;
        this->gradient = &Relu_gradient;
        goto ENDACTIVATON;
    }
    if (mystrcmp(activation, "softmax") == 0)
    {
        this->activation = &Softmax;
        goto ENDACTIVATON;
    }
    if (mystrcmp(activation, "sigmoid") == 0)
    {
        this->activation = &Sigmoid;
        this->gradient = &Sigmoid_gradient;
        goto ENDACTIVATON;
    }
    if (mystrcmp(activation, "leakyrelu") == 0)
    {
        this->activation = &LeakyRelu;
        this->gradient = &LeakyRelu_gradient;
        goto ENDACTIVATON;
    }
    if (mystrcmp(activation, "linear") == 0)
    {
        this->activation = &Linear;
        this->gradient = &Linear_gradient;
        goto ENDACTIVATON;
    }
ENDACTIVATON:
    //Initialize weight matrix and bias vector
    MYTYPE t =  sqrt(6.0 / (input_units + output_units));
    xavier = std::uniform_real_distribution<MYTYPE>(-t, t);
    for (int i = 0; i < weight.rows(); i++)
        for (int j = 0; j < weight.cols(); j++)
        {
            weight(i, j) = xavier(dense_weight);
        }
    weight.DataTransfer(HostToDevice);
    for (int i = 0; i < bias.size(); i++)
        bias[i] = 0.0;
    bias.DataTransfer(HostToDevice);

    return true;
}

void Dense::Forward(Dense* pre_layer)
{
    //cudaError_t status;
    if (!pre_layer)
    {
        printf("ERROR: pre_layer is null pointer.\n");
        getchar();
        return;
    }

    int threads = (512);
    int blocks = ((input_units + threads - 1) / threads);

    //z = Wx + b
    mc.gemv_gpu(weight.GetDevMat(), weight.rows(), weight.cols(), pre_layer->output.GetDevVec(), output.GetDevVec());

    if (activation != &Softmax)
    {
        output = output + bias;
        //a = activation(z)
        activation <<<blocks, threads >>> (output.GetDevVec(), output.GetDevVec(), output_units);
        cudaDeviceSynchronize();
        output.DataTransfer(DeviceToHost);
    }
    else//Softmax works on CPU
    {
        output = output + bias;
        output.DataTransfer(DeviceToHost);
        activation(output.GetVec(), output.GetVec(), output_units);
        output.DataTransfer(HostToDevice);
    }
}

void Dense::Forward(Vector& _input, const int element_num)
{
    if (_input.empty())
    {
        printf("ERROR: input vector _input is empty.\n");
        getchar();
        return;
    }
    
    int threads = (512);
    int blocks = ((output.size() + threads - 1) / threads);

    //z = Wx + b
    mc.gemv_gpu(weight.GetDevMat(), weight.rows(), weight.cols(), _input.GetDevVec(), output.GetDevVec());
    output = output + bias;

    if (activation != Softmax)
    {
        activation <<<blocks, threads >>> (output.GetDevVec(), output.GetDevVec(), output_units);
        cudaDeviceSynchronize();
    }
    else//Softmax works on CPU
    {
        output.DataTransfer(DeviceToHost);
        activation(output.GetVec(), output.GetVec(), output_units);
        output.DataTransfer(HostToDevice);
    }
    
    output.DataTransfer(DeviceToHost);
}

void Dense::Backward(Vector& _loss, Dense* pre_layer, const int num)
{
    if (_loss.empty() || _loss.size() != num || num <= 0)
    {
        printf("ERROR: invalide parament(s). Please double check whether _loss.size()==num, num > 0, _loss is not empty.\n");
        getchar();
        return;
    }
    if (!pre_layer)
    {
        printf("ERROR: parament pre_layer is null pointer.\n");
        getchar();
        return;
    }

    //Transposition weight matrix
    for (int i = 0; i < weight_t.rows(); i++)
        for (int j = 0; j < weight_t.cols(); j++)
            weight_t(i, j) = weight(j, i);
    weight_t.DataTransfer(HostToDevice);

    /*
    In BP, we will update weight matrix and bias vector
    */
    //For the last layer, the gradient of Softmax + Cross Entropy already calculated in Net::Backward()
    //so calculate delta = loss * x directly
    mc.gemm_gpu(_loss.GetDevVec(), num, 1, pre_layer->output.GetDevVec(), 1, pre_layer->output.size(), weight_grad.GetDevMat(), weight_grad.rows());

    //loss in previous layer delta' = transposition(W) * loss
    mc.gemv_gpu(weight_t.GetDevMat(), weight_t.rows(), weight_t.cols(), _loss.GetDevVec(), pre_layer->loss.GetDevVec());
    pre_layer->loss.DataTransfer(DeviceToHost);

    //gradient = lr * loss
    mc.MatrixMultNumber(weight_grad.GetDevMat(), lr, weight_grad.rows(), weight_grad.cols());
    mc.VecMultNum(_loss.GetDevVec(), lr, _loss.size());

    cudaMemcpy(temp_grad, weight_grad.GetDevMat(), sizeof(MYTYPE) * output_units * input_units, cudaMemcpyDeviceToHost);

    //Matrix weight_grad is column-major matrix, while matrix weight is a row-major matrix whose data copied from CPU directly.
    //Running code weight - weight_grad without reforming weight_grad will cause mistakes.
    Colmaj2Rowmaj(temp_grad, weight_grad.GetMat(), weight_grad.rows(), weight_grad.cols());
    weight_grad.DataTransfer(HostToDevice);
    weight -= weight_grad;
    weight.DataTransfer(DeviceToHost);

    bias = bias - _loss;
    this->loss = _loss;
}

void Dense::Backward(Dense* pre_layer)
{
    Vector differal(output_units);
    if (!pre_layer)
    {
        printf("ERROR: parament pre_layer is null pointer.\n");
        getchar();
        return;
    }

    for (int i = 0; i < weight_t.rows(); i++)
        for (int j = 0; j < weight_t.cols(); j++)
            weight_t(i, j) = weight(j, i);
    weight_t.DataTransfer(HostToDevice);


    //gradient = loss * f'(a) * x
    int threads = (512);
    int blocks = ((output.size() + threads - 1) / threads);

    //calculate f'(a)
    gradient<<<blocks,threads>>>(output.GetDevVec(), differal.GetDevVec(), output_units);
    cudaDeviceSynchronize();

    //loss * f'(a) 

    mc.VecEleMult(loss.GetDevVec(), differal.GetDevVec(), output_units);

    //calculate loss value in previous layer
    mc.gemv_gpu(weight_t.GetDevMat(), weight_t.rows(), weight_t.cols(), loss.GetDevVec(), pre_layer->loss.GetDevVec());
    loss.DataTransfer(DeviceToHost);

    //calculate graident
    mc.gemm_gpu(loss.GetDevVec(), output_units, 1, pre_layer->output.GetDevVec(), 1, pre_layer->output_units,
        weight_grad.GetDevMat(), weight_grad.rows());

    mc.MatrixMultNumber(weight_grad.GetDevMat(), lr, weight_grad.rows(), weight_grad.cols());

    mc.VecMultNum(loss.GetDevVec(), lr, loss.size());

    cudaMemcpy(temp_grad, weight_grad.GetDevMat(), sizeof(MYTYPE) * output_units * input_units, cudaMemcpyDeviceToHost);

    //Matrix weight_grad is column-major matrix, while matrix weight is a row-major matrix whose data copied from CPU directly.
    //Running code weight - weight_grad without reforming weight_grad will cause mistakes.
    Colmaj2Rowmaj(temp_grad, weight_grad.GetMat(), weight_grad.rows(), weight_grad.cols());
    weight_grad.DataTransfer(HostToDevice);
    weight -= weight_grad;

    bias -= loss;
    weight.DataTransfer(DeviceToHost);    

}

void Dense::Backward(Flatten* pre_layer)
{
    Vector differal(output_units);
    if (!pre_layer)
    {
        printf("ERROR: pre_layer (Flatten*) in Dense::Backward is null pointer.\n");
        getchar();
        return;
    }

    for (int i = 0; i < weight_t.rows(); i++)
        for (int j = 0; j < weight_t.cols(); j++)
            weight_t(i, j) = weight(j, i);
    weight_t.DataTransfer(HostToDevice);


    int threads = (512);
    int blocks = ((output.size() + threads - 1) / threads);

    //calculate f'(a)
    gradient <<<blocks, threads >>> (output.GetDevVec(), differal.GetDevVec(), output_units);
    cudaDeviceSynchronize();

    //loss * f'(a)
    mc.VecEleMult(loss.GetDevVec(), differal.GetDevVec(), output_units);

    //calculate loss value in previous layer
    mc.gemv_gpu(weight_t.GetDevMat(), weight_t.rows(), weight_t.cols(), loss.GetDevVec(), pre_layer->loss.GetDevVec());
    loss.DataTransfer(DeviceToHost);

    //calculate graident
    mc.gemm_gpu(loss.GetDevVec(), loss.size(), 1, pre_layer->GetOutput().GetDevVec(), 1, pre_layer->GetSize(),
        weight_grad.GetDevMat(), weight_grad.rows());

    mc.MatrixMultNumber(weight_grad.GetDevMat(), lr, weight_grad.rows(), weight_grad.cols());
    mc.VecMultNum(loss.GetDevVec(), lr, loss.size());
    weight_grad.DataTransfer(DeviceToHost);

    //Matrix weight_grad is column-major matrix, while matrix weight is a row-major matrix whose data copied from CPU directly.
    //Running code weight - weight_grad without reforming weight_grad will cause mistakes.
    cudaMemcpy(temp_grad, weight_grad.GetDevMat(), sizeof(MYTYPE) * output_units * input_units, cudaMemcpyDeviceToHost);
    Colmaj2Rowmaj(temp_grad, weight_grad.GetMat(), weight_grad.rows(), weight_grad.cols());
    weight_grad.DataTransfer(HostToDevice);

    weight -= weight_grad;
    bias -= loss;

    weight.DataTransfer(DeviceToHost);
}

void Dense::Backward(Vector& _input)
{
    Vector differal(output_units);
    if (_input.empty())
    {
        printf("ERROR: empty vector _input in Dense::Backward(Vector&).\n");
        getchar();
        return;
    }

    for (int i = 0; i < weight_t.rows(); i++)
        for (int j = 0; j < weight_t.cols(); j++)
            weight_t(i, j) = weight(j, i);
    weight_t.DataTransfer(HostToDevice);

    int threads = (512);
    int blocks = ((output.size() + threads - 1) / threads);

    //calculate f'(a)
    gradient << <blocks, threads >> > (output.GetDevVec(), differal.GetDevVec(), output_units);
    cudaDeviceSynchronize();

    //loss * f'(a)
    mc.VecEleMult(loss.GetDevVec(), differal.GetDevVec(), output_units);

    //calculate graident
    mc.gemm_gpu(loss.GetDevVec(), output_units, 1, _input.GetDevVec(), 1,_input.size(),
        weight_grad.GetDevMat(), weight_grad.rows(), CUBLAS_OP_T, CUBLAS_OP_T);
    weight_grad.DataTransfer(DeviceToHost);
    
    mc.MatrixMultNumber(weight_grad.GetDevMat(), lr, weight_grad.rows(), weight_grad.cols());
    mc.VecMultNum(loss.GetDevVec(), lr, loss.size());

    //Matrix weight_grad is column-major matrix, while matrix weight is a row-major matrix whose data copied from CPU directly.
    //Running code weight - weight_grad without reforming weight_grad will cause mistakes.
    cudaMemcpy(temp_grad, weight_grad.GetDevMat(), sizeof(MYTYPE) * output_units * input_units, cudaMemcpyDeviceToHost);
    Colmaj2Rowmaj(temp_grad, weight_grad.GetMat(), weight_grad.rows(), weight_grad.cols());
    weight_grad.DataTransfer(HostToDevice);
    weight -= weight_grad;
    bias -= loss;

    weight.DataTransfer(DeviceToHost);
}

void Dense::Save(string& _dir, int which)
{
    FILE* fp = fopen((_dir + "/" + "Dense" + std::to_string(which) + ".txt").c_str(), "w");
    if (!fp)
    {
        printf("Cannot open file %s\n", (_dir + "/" + "Dense" + std::to_string(which) + ".txt").c_str());
        getchar();
        return;
    }

    fprintf(fp, "lr=%f\n", lr);

    for (int i = 0; i < weight.rows(); i++)
    {
        for (int j = 0; j < weight.cols(); j++)
            fprintf(fp, "%.7f ", weight(i, j));
        fprintf(fp, "\n");
    }

    fprintf(fp, "bias:\n");
    bias.DataTransfer(DeviceToHost);
    for (int i = 0; i < bias.size(); i++)
        fprintf(fp, "%f\n", bias[i]);
    
    weight_grad.DataTransfer(DeviceToHost);
    fprintf(fp, "grad:\n");
    for (int i = 0; i < weight_grad.rows(); i++)
    {
        for (int j = 0; j < weight_grad.cols(); j++)
            fprintf(fp, "%.10f ", weight_grad(i, j));
        fprintf(fp, "\n");
    }
    fprintf(fp, "loss:\n");
    for (int i = 0; i < loss.size(); i++)
    {
        fprintf(fp, "%.10f ", loss[i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void Dense::lrDecay(const int now_epoch)
{
    lr = lr * pow(0.98, now_epoch);
}