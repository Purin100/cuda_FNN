#include "Dense.cuh"
#include <ctime>

static std::bernoulli_distribution ber(0.5);
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

bool Dense::BuildLayer(int input_units, int output_units, const char* activation, const char* optimizer, bool isfreeze)
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

    this->input_units = input_units;
    this->output_units = output_units;
    this->freeze_weight = isfreeze;

    weight = Matrix(output_units, input_units);
    weight_grad = Matrix(output_units, input_units);
    grad_sample = Matrix(output_units, input_units);
    weight_t = Matrix(input_units, output_units);
    save_grad = Matrix(output_units, input_units);

    grad_direction = Matrix(output_units, input_units);

    output = Vector(output_units);
    loss = Vector(output_units);
    bias = Vector(output_units);
    bias_batch = Vector(output_units);
        
    local_out = Vector(output_units);
    input = Vector(input_units);

    //Set activation function and relevant gradient function
    //Softmax DO NOT have gradient function, and cannot be used in hidden layers
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
    std::uniform_real_distribution<MYTYPE> xavier(-t, t);
    for (int i = 0; i < weight.rows(); i++)
        for (int j = 0; j < weight.cols(); j++)
        {
            weight(i, j) = xavier(dense_weight);
        }
    weight.DataTransfer(HostToDevice);
    for (int i = 0; i < bias.size(); i++)
        bias[i] = xavier(dense_weight);
    bias.DataTransfer(HostToDevice);
#ifdef lr_mat
    for (int i = 0; i < output_units; i++)
        for (int j = 0; j < input_units; j++)
            lr(i, j) = lr_dis(dense_weight);
    lr.DataTransfer(HostToDevice);
#endif
    if ((optimizer) && mystrcmp(optimizer, "adam") == 0)
        if (!adam.Init(output_units, input_units, "matrix"))
            return false;
    weight_grad.Zeroreset();
    bias_batch.ZeroReset();
    save_grad.Zeroreset();
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
;
    int threads = 32;
    int blocks;

    //z = Wx + b
    mc.gemv_gpu(weight.GetDevMat(), weight.rows(), weight.cols(), pre_layer->output.GetDevVec(), local_out.GetDevVec());
    local_out += bias;
   
    blocks = (output_units + threads - 1) / threads;
    if (activation != &Softmax)
    {
        //a = activation(z)
        activation <<<blocks, threads >>> (local_out.GetDevVec(), output.GetDevVec(), output_units);
        cudaDeviceSynchronize();
        output.DataTransfer(DeviceToHost);
    }
    else//Softmax works on CPU
    {
        local_out.DataTransfer(DeviceToHost);
        activation(local_out.GetVec(), output.GetVec(), output_units);
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
    
    int threads = 32;
    int blocks;

    blocks = (output.size() + threads - 1) / threads;
    //z = Wx + b
    //mc.gemv_gpu(weight.GetDevMat(), weight.rows(), weight.cols(), _input.GetDevVec(), output.GetDevVec());
    mc.gemv_gpu(weight.GetDevMat(), output_units, input_units, _input.GetDevVec(), local_out.GetDevVec());
    local_out += bias;   

    if (activation != Softmax)
    {
        activation << <blocks, threads >> > (local_out.GetDevVec(), output.GetDevVec(), output_units);
        cudaDeviceSynchronize();
        output.DataTransfer(DeviceToHost);
    }
    else//Softmax works on CPU
    {
        local_out.DataTransfer(DeviceToHost);
        activation(local_out.GetVec(), output.GetVec(), output_units);
        output.DataTransfer(HostToDevice);
    }   
}

void Dense::Backward(Vector& _loss, Dense* pre_layer, bool update)
{
    if (_loss.empty())
    {
        printf("ERROR: invalide parament(s). Please double check whether batch_size > 0, _loss is not empty.\n");
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
    mc.MatrixTranspose(weight.GetDevMat(), weight_t.GetDevMat(), output_units, input_units);
    //weight_t.DataTransfer(DeviceToHost);


    /*
    In BP, we will update weight matrix and bias vector
    */
    bias_batch += _loss;
    //loss in previous layer delta' = transposition(W) * loss
    mc.gemv_gpu(weight_t.GetDevMat(), input_units, output_units, _loss.GetDevVec(), pre_layer->loss.GetDevVec());

    if (freeze_weight)
        return;
    //For the last layer, the gradient of Softmax + Cross Entropy already calculated in Net::Backward()
    //so calculate delta = loss * x directly
    //mc.gemm_gpu(_loss.GetDevVec(), output_units, 1, pre_layer->output.GetDevVec(), 1, input_units,
    //    grad_sample.GetDevMat(), output_units, CUBLAS_OP_N, CUBLAS_OP_N);
    mc.gemm_gpu(_loss.GetDevVec(), output_units, 1, pre_layer->output.GetDevVec(), 1, input_units,
        grad_sample.GetDevMat(), output_units, CUBLAS_OP_N, CUBLAS_OP_N);
    weight_grad += grad_sample;
    
    sample_num++;

    if (update)
    {
        //gradient = lr * loss
        if (adam.InitState())
        {
            adam.Update((weight_grad), weight);
        }
        else
            weight -= weight_grad * lr;
        //weight_grad.DataTransfer(DeviceToHost);
        //bias -= _loss * lr;
        bias -= bias_batch * lr;
        save_grad = weight_grad;
        weight_grad.Zeroreset();
        bias_batch.ZeroReset();
        sample_num = 0;
    }
    this->loss = _loss;
}

void Dense::Backward(Dense* pre_layer, bool update)
{
#ifdef DENSE_WEIGHTOUT
    Vector differal(input_units);
#else
    Vector differal(output_units);
#endif
    if (!pre_layer)
    {
        printf("ERROR: parament pre_layer is null pointer.\n");
        getchar();
        return;
    }

    mc.MatrixTranspose(weight.GetDevMat(), weight_t.GetDevMat(), output_units, input_units);
    //weight_t.DataTransfer(DeviceToHost);

    //gradient = loss * f'(a) * x
    int threads = 32;
    int blocks;

    blocks = (output_units + threads - 1) / threads;
    //calculate f'(a)
    //gradient<<<blocks,threads>>>(output.GetDevVec(), differal.GetDevVec(), output_units);
    gradient <<<blocks, threads >>> (local_out.GetDevVec(), differal.GetDevVec(), output_units);
    cudaDeviceSynchronize();

    bias_batch += loss;
    //loss * f'(a) 
    // loss value here should multiply with differal before update weights in this layer
    mc.VecEleMult(loss.GetDevVec(), differal.GetDevVec(), output_units);

    //calculate loss value in previous layer
    mc.gemv_gpu(weight_t.GetDevMat(), input_units, output_units, loss.GetDevVec(), pre_layer->loss.GetDevVec());

    if (freeze_weight)
        return;
    //calculate graident
    mc.gemm_gpu(loss.GetDevVec(), output_units, 1, pre_layer->output.GetDevVec(), 1, input_units,
        grad_sample.GetDevMat(), output_units, CUBLAS_OP_N, CUBLAS_OP_N);
    weight_grad += grad_sample;
    
    sample_num++;

    if (update)
    {
        if (adam.InitState())
            adam.Update(weight_grad, weight);
        else
            weight -= weight_grad * lr;
        bias -= bias_batch * lr;
        save_grad = weight_grad;
        weight_grad.Zeroreset();
        bias_batch.ZeroReset();
        sample_num = 0;
    }
}

void Dense::Backward(Flatten* pre_layer, bool update)
{
    Vector differal(output_units);
    if (!pre_layer)
    {
        printf("ERROR: pre_layer (Flatten*) in Dense::Backward is null pointer.\n");
        getchar();
        return;
    }

    mc.MatrixTranspose(weight.GetDevMat(), weight_t.GetDevMat(), output_units, input_units);
    //weight_t.DataTransfer(DeviceToHost);

    int threads = 32;
    int blocks;

    blocks = (output_units + threads - 1) / threads;
    //calculate f'(a)
    gradient <<<blocks, threads >>> (local_out.GetDevVec(), differal.GetDevVec(), output_units);
    cudaDeviceSynchronize();

    if(!freeze_weight)
        bias_batch += loss;
    //loss * f'(a)
    mc.VecEleMult(loss.GetDevVec(), differal.GetDevVec(), output_units);

    //calculate loss value in previous layer
    mc.gemv_gpu(weight_t.GetDevMat(), weight_t.rows(), weight_t.cols(), loss.GetDevVec(), pre_layer->loss.GetDevVec());

    if (freeze_weight)
        return;
    //calculate graident
    mc.gemm_gpu(loss.GetDevVec(), output_units, 1, pre_layer->GetOutput().GetDevVec(), 1, pre_layer->GetSize(),
        grad_sample.GetDevMat(), output_units, CUBLAS_OP_N, CUBLAS_OP_N);
    weight_grad += grad_sample;
    
    sample_num++;

    if (update)
    {
        if (adam.InitState())
            adam.Update(weight_grad, weight);
        else
            weight -= weight_grad * lr;
        //bias -= loss * lr;
        bias -= bias_batch * lr;
        save_grad = weight_grad;
        weight_grad.Zeroreset();
        bias_batch.ZeroReset();
        sample_num = 0;
    }

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

    mc.MatrixTranspose(weight.GetDevMat(), weight_t.GetDevMat(), output_units, input_units);

    int threads = 32;
    int blocks = ((output.size() + threads - 1) / threads);

    //calculate f'(a)
    gradient <<<blocks, threads >>> (local_out.GetDevVec(), differal.GetDevVec(), output_units);
    cudaDeviceSynchronize();

    //loss * f'(a)
    loss.DataTransfer(DeviceToHost);
    mc.VecEleMult(loss.GetDevVec(), differal.GetDevVec(), output_units);

    //calculate loss value in previous layer
    //mc.gemv_gpu(weight_t.GetDevMat(), weight_t.rows(), weight_t.cols(), loss.GetDevVec(), pre_layer->loss.GetDevVec());
    //loss.DataTransfer(DeviceToHost);

    //calculate graident
    mc.gemm_gpu(loss.GetDevVec(), output_units, 1, _input.GetDevVec(), 1,_input.size(),
        weight_grad.GetDevMat(), weight_grad.rows(), CUBLAS_OP_N, CUBLAS_OP_N);
    weight_grad.DataTransfer(DeviceToHost);
    

    weight -= weight_grad * lr;
    bias -= loss * lr;
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
    
    fprintf(fp, "grad:\n");
    for (int i = 0; i < output_units; i++)
    {
        for (int j = 0; j < input_units; j++)
            fprintf(fp, "%.10f ", save_grad(i, j));
        fprintf(fp, "\n");
    }


    fprintf(fp, "loss:\n");
    loss.DataTransfer(DeviceToHost);
    for (int i = 0; i < loss.size(); i++)
    {
        fprintf(fp, "%.10f ", loss[i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void Dense::lrDecay(const int now_epoch)
{
    lr *= pow(0.98, now_epoch);
}