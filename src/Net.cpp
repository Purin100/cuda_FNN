#include "Net.h"

Net::Net(const int samples, const int batch_size)
{
    assert(samples > 0);
    this->batch_size = batch_size > 0 ? batch_size : 1;
    if (samples % batch_size == 0)
        batches = samples / batch_size;
    else
    {
        batches = samples / batch_size + 1;
        rest_sample = samples % batch_size;
    }
}

Net::~Net()
{

}

void Net::Forward(MYTYPE* _input, const int size)
{
    Dense* dlayer_now = nullptr, * dlayer_pre = nullptr;
    Flatten* flayer_now = nullptr;

    if (!_input)
    {
        printf("ERROR: empty pointer _image.\n");
        getchar();
        abort();
    }

    if (input.empty())
        input = Vector(_input, size);

    auto now_layer = layers.begin();
    auto pre_layer = layers.begin();

    if (now_layer->type == DENSE)
        reinterpret_cast<Dense*>(now_layer->layer)->Forward(input, size);
    now_layer++;

    int preType;
    while (now_layer < layers.end())
    {
        preType = pre_layer->type;

        switch (now_layer->type)
        {
        case DENSE:
            dlayer_now = (Dense*)((*now_layer).layer);
            if (preType == DENSE)
            {
                dlayer_pre = reinterpret_cast<Dense*>(pre_layer->layer);
                dlayer_now->Forward(dlayer_pre);
            }
            break;
        }

        now_layer++;
        pre_layer++;
    }
}

void Net::Forward(TXTReader* _input)
{
    if (!_input)
    {
        printf("ERROR: parament _input is null pointer.\n");
        getchar();
        return;
    }
    Dense* dlayer_now = nullptr, * dlayer_pre = nullptr;
    Flatten* flayer_now = nullptr;

    auto now_layer = layers.begin();
    auto pre_layer = layers.begin();

    if (now_layer->type == FLATTEN)
        reinterpret_cast<Flatten*>(now_layer->layer)->Forward(_input->Getdata(0), _input->Width(), _input->Height());
    now_layer++;

    int preType;
    while (now_layer < layers.end())
    {
        preType = pre_layer->type;

        switch (now_layer->type)
        {
        case DENSE:
            dlayer_now = reinterpret_cast<Dense*>((*now_layer).layer);
            if (preType == DENSE)
            {
                dlayer_pre = reinterpret_cast<Dense*>(pre_layer->layer);
                dlayer_now->Forward(dlayer_pre);
            }
            if (preType == FLATTEN)
            {
                dlayer_now->Forward(reinterpret_cast<Flatten*>(pre_layer->layer)->GetOutput(),
                    reinterpret_cast<Flatten*>(pre_layer->layer)->GetSize());
            }
            break;
        }

        now_layer++;
        pre_layer++;
    }
}

void Net::Forward(Vector& _input, int row, int col)
{
    Dense* dlayer_now = nullptr, * dlayer_pre = nullptr;
    Flatten* flayer_now = nullptr;
    auto now_layer = layers.begin();
    auto pre_layer = layers.begin();

    if (now_layer->type == FLATTEN)
        reinterpret_cast<Flatten*>(now_layer->layer)->Forward(_input);
    now_layer++;

    int preType;
    while (now_layer < layers.end())
    {
        preType = pre_layer->type;

        switch (now_layer->type)
        {
        case DENSE:
            dlayer_now = (Dense*)((*now_layer).layer);
            if (preType == DENSE)
            {
                dlayer_pre = reinterpret_cast<Dense*>(pre_layer->layer);
                dlayer_now->Forward(dlayer_pre);
            }
            if (preType == FLATTEN)
            {
                dlayer_now->Forward(reinterpret_cast<Flatten*>(pre_layer->layer)->GetOutput(),
                    reinterpret_cast<Flatten*>(pre_layer->layer)->GetSize());
            }
            break;
        case FLATTEN:
            flayer_now = (Flatten*)((*now_layer).layer);
            break;
        }

        now_layer++;
        pre_layer++;
    }
}

void Net::Backward(Vector onehot_label)
{
    if (onehot_label.empty())
    {
        printf("ERROR: parament onehot_label is empty.\n");
        getchar();
        return;
    }

    int category = layers.back().num;
    Vector last_out = GetLayerOutput(layers.back().order);


    loss = Cross_Entropy(onehot_label.GetVec(), last_out.GetVec(), category);
    //loss = MSE(onehot_label.GetVec(), last_out.GetVec(), category);
    total_loss += loss;
    //printf("loss in current sample: %f\n", loss);

    //gradient for Softmax function
    last_out -= onehot_label;

    this->current_sample++;
    this->update = false;

    //if reaches the batch size, then allow updating weight
    if (this->current_sample == batch_size)
    {
        this->update = true;
        current_batch++;
        this->current_sample = 0;
    }
    //or when comes to the last batch, allow updating weight
    if (rest_sample && this->current_batch == batches)
    {
        if (current_sample == rest_sample)
        {
            this->update = true;
            this->current_sample = 0;
        }
    }


    //BP starts here
    auto now_layer = layers.end() - 1;
    auto pre_layer = layers.end() - 2;

    while (now_layer >= layers.begin())
    {
        //The last layer has different inputs, so it will be a seperate branch
        if (now_layer == layers.end() - 1)
        {
            if (now_layer->type == DENSE && pre_layer->type == DENSE)
            {
                reinterpret_cast<Dense*>(now_layer->layer)->Backward(last_out, reinterpret_cast<Dense*>(pre_layer->layer), this->update);
            }
            now_layer--;
            if (pre_layer != layers.begin())
                pre_layer--;
            continue;
        }

        //Other layers' BP
        switch (now_layer->type)
        {
        case DENSE:
            if (pre_layer->type == DENSE)
            {
                if (now_layer == layers.begin())
                    reinterpret_cast<Dense*>(now_layer->layer)->Backward(input);
                else
                    reinterpret_cast<Dense*>(now_layer->layer)->Backward(reinterpret_cast<Dense*>(pre_layer->layer), this->update);
            }
            if (pre_layer->type == FLATTEN)
            {
                reinterpret_cast<Dense*>(now_layer->layer)->Backward(reinterpret_cast<Flatten*>(pre_layer->layer), this->update);
            }
            break;
        default:
            break;
        }

        now_layer--;
        if (pre_layer != layers.begin())
            pre_layer--;
        else//If pre_layer == layers.begin(), BP ends.
            break;
    }
}

bool Net::Eval(int label, Vector onehot_label)
{
    MYTYPE max = -99.0;
    int predict = -1;
    int category = layers.back().num;
    Vector last_out = GetLayerOutput(layers.back().order);

    //calculate loss value
    loss = Cross_Entropy(onehot_label.GetVec(), last_out.GetVec(), category);
    total_loss += loss;


    for (int i = 0; i < category; i++)
    {
        if (last_out[i] > max)
        {
            max = last_out[i];
            predict = i;
        }
    }
    if (predict >= 0 && predict < category)
        cate_res[predict]++;
    else
    {
        printf("ERROR: invalid predict result!\n");
        getchar();
        exit(2);
    }
    confuse[label][ predict]++;
    return (predict == label);
}

void Net::Save(std::string _dir)
{
//create floders for saving
#if defined(_WIN32) || defined(_WIN64)
    if (_access(_dir.c_str(), 0) != 0)
        if (_mkdir(_dir.c_str()) != 0)
        {
            printf("Create directory failed.\n");
            return;
        }
#endif
#ifdef linux
    if (access(_dir.c_str(), 0) != 0)
        if (mkdir(_dir.c_str(), 0777) < 0)
        {
            printf("Create directory failed.\n");
            return;
        }
#endif

    auto layer = layers.begin();
    while (layer < layers.end())
    {
        switch (layer->type)
        {
        case DENSE:
            reinterpret_cast<Dense*>(layer->layer)->Save(_dir, layer->order);
            break;
        default:
            break;
        }
        layer++;
    }

    FILE* fp = fopen((_dir + "/confuse matrix.txt").c_str(), "w");
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 10; j++)
            fprintf(fp, "%d ", confuse[i][j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
}

const Vector& Net::GetLayerOutput(int which)
{
    
    if (which<0 || which>order)
    {
        printf("ERROR: parament which is not in the range of [0, %d).\n", order);
        getchar();
        abort();
    }
    if (layers[which].type == DENSE)
        return reinterpret_cast<Dense*>(layers[which].layer)->Getoutput();
    
}

void Net::lrDecay(const int now_epoch)
{
    if (layers.empty())
        return;
    auto now_layer = layers.begin();

    while (now_layer < layers.end())
    {
        if (now_layer->type == DENSE)
            reinterpret_cast<Dense*>(now_layer->layer)->lrDecay(now_epoch);
        now_layer++;
    }
}
