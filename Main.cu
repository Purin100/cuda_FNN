#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Matrix.cuh"
#include "utils.cuh"
#include "Activation.cuh"
#include "Dense.cuh"
#include "Net.h"
#include "TXTReader.h"
#include <stdio.h>
#if defined(linux) || defined(Liunx)
#include <dirent.h>
#endif

#define CLOSEFILE(fp){if(fp) fclose(fp);}
//List files with specific extension name ext in one floder
void Listdir(std::string _dir, const char* ext, std::vector<string>& result);

struct Dataset
{
    Dataset(TXTReader* _file, int _label)
    {
        file = _file, label = _label;
    }
    TXTReader* file;
    int label;
};

int main(int argc, char** argv)
{
    TXTReader* file, * valid;
    
#ifdef _DEBUG
    int trainFile_num = 60000, testFile_num = 10000;
#else
    int trainFile_num = 200, testFile_num = 200;
#endif
    int obj_label;

    int* train_label_arr = nullptr, * test_label_arr = nullptr;
    int epoch = 500;//total training epoch
    int now_epoch = 0;
    std::vector<string> trainfiles, testfiles;

    //Change directory to your own directory  
    Listdir(std::string("./trainsamples"), ".txt", trainfiles);
    Listdir(std::string("./testsamples"), ".txt", testfiles);


    //read training data and test data from files
    //trainFile_num = trainfiles.size();
    //testFile_num = testfiles.size();
    file = new TXTReader[trainFile_num];
    valid = new TXTReader[testFile_num];

    for (int i = 0; i < trainFile_num; i++)
    {
        file[i].ReadFile("mnist", trainfiles[i]);
        file[i].Shrink(MINUS_ONE_TO_ONE);
    }
    for (int i = 0; i < testFile_num; i++)
    {
        valid[i].ReadFile("mnist", testfiles[i]);
        valid[i].Shrink(MINUS_ONE_TO_ONE);
    }

    //Change directory to your own directory
    FILE* train_label = fopen("./train_label.txt", "r");
    if (!train_label)
    {
        printf("ERROR: Open train_label failed.\n");
        return -1;
    }

    //Change directory to your own directory
    FILE* test_label = fopen("./test_label.txt", "r");
    //FILE* test_label = fopen("D:/CudaRun/0train.txt", "r");
    if (!test_label)
    {
        printf("ERROR: Open test_label failed.\n");
        fclose(train_label);
        return -1;
    }

    train_label_arr = new int[trainFile_num];
    test_label_arr = new int[testFile_num];

    for (int i = 0; i < testFile_num; i++)
        fscanf(test_label, "%d", &test_label_arr[i]);
    for (int i = 0; i < trainFile_num; i++)
        fscanf(train_label, "%d", &train_label_arr[i]);
    printf("label loaded.\n");

    fclose(train_label);
    fclose(test_label);
    printf("File load complete.\n");

    std::vector<Dataset> trainset, testset;
    for (int i = 0; i < trainFile_num; i++)
        trainset.push_back(Dataset(&file[i], train_label_arr[i]));
    for (int i = 0; i < testFile_num; i++)
        testset.push_back(Dataset(&valid[i], test_label_arr[i]));

    if (argc == 2)
        epoch = atoi(argv[1]);

    //Declear layer objects
    Dense d, d1, df;
    Flatten flatten;
    FlattenInfo fi(28, 28);
    cudaError_t cudaStatus;

    //one-hot label matrix
    Matrix one_hot = Identity(10);

    //add layers to the network
    Net net;
    net.add(&flatten, &fi);
    net.add(&d1, 128,1.0,"tanh");
    net.add(&df, 10,1.0,"softmax");

    int count = 0;
    MYTYPE* loss = new MYTYPE[epoch];
    MYTYPE* accuracy = new MYTYPE[epoch]{ 0.0 };
    MYTYPE* tloss = new MYTYPE[epoch];
    FILE* f;

    //main loop
    while (now_epoch < epoch)
    {
        count = 0;
        net.train = true;

        //shuffle training set
        std::random_shuffle(trainset.begin(), trainset.end());

        //input one file each time
        while (count < trainFile_num * 0.8)//80% of the training files use for training
        {
            obj_label = trainset[count].label;
            net.Forward(trainset[count].file);
            net.Backward(one_hot.RowSlice(obj_label));
            //net.Save(std::to_string(now_epoch) + std::to_string(count));
            count++;
        }
        printf("Loss in %d epoch: %f\n", now_epoch, net.total_loss / (trainFile_num * 0.8));

        //save weights every epoch
        net.Save(std::to_string(now_epoch));
        //loss[now_epoch] = net.total_loss / trainFile_num;

        //decrease the learning rate in the network
        //if (now_epoch % 5 == 0)
            //net.lrDecay(now_epoch);

        //reset total_loss, this variable will record loss in validation process
        net.total_loss = 0.0;

        while(count < trainFile_num )//20% of the training files use for validation
        {
            obj_label = trainset[count].label;
            net.Forward(trainset[count].file);
            if (net.Eval(obj_label, one_hot.RowSlice(obj_label)))
                accuracy[now_epoch] += 1.0;
            count++;
        }
        accuracy[now_epoch] /= (trainFile_num * 0.2);
        tloss[now_epoch] = net.total_loss / (trainFile_num * 0.2);
        printf("Valid accuracy in %d epoch: %f, loss in this epoch: %f\n", now_epoch, accuracy[now_epoch], tloss[now_epoch]);
        //f = fopen(string(std::to_string(now_epoch) + "/cate_res.txt").c_str(), "w");
        //for (int i = 0; i < 10; i++)
        //    fprintf(f, "%d ", net.cate_res[i]);
        //fclose(f);
        for (int i = 0; i < 10; i++)
            printf("%d ", net.cate_res[i]);
        printf("\n");
        memset(net.cate_res, 0, sizeof(int) * 10);

        net.total_loss = 0.0;
        now_epoch++;
    }

    //test process
    count = 0;
    accuracy[0] = 0.0;
    net.total_loss = 0.0;
    while (count < testFile_num)
    {
        obj_label = test_label_arr[count];
        net.Forward(&valid[count]);
        if (net.Eval(obj_label, one_hot.RowSlice(obj_label)))
            accuracy[0] += 1.0;
        count++;
    }
    accuracy[0] /= testFile_num;
    printf("Test accuracy in %d epoch: %f, loss in this epoch: %f\n\n", now_epoch, accuracy[0], net.total_loss / testFile_num);
    f = fopen("./cate_res.txt", "w");
    for (int i = 0; i < 10; i++)
        fprintf(f, "%d ", net.cate_res[i]);
    fclose(f);
    memset(net.cate_res, 0, sizeof(int) * 10);

    //Destory cuda handle before reset the device
    mc.Destory();
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    //free memcory
    RELEASE(loss);
    RELEASE(tloss);
    RELEASE(file);
    RELEASE(valid);
    RELEASE(train_label_arr);
    RELEASE(test_label_arr);
    RELEASE(accuracy);
    return 0;
}

#if defined(liunx) || defined(Linux)
void Listdir(std::string _dir, const char* ext, std::vector<string>& result)
{
    DIR* dp;
    struct dirent* dirp;
    std::string temp;
    int ext_len, len;
    if (_dir.empty())
    {
        printf("ERROR: empty directory %s\n", _dir.c_str());
        getchar();
        return;
    }

    if (!(dp = opendir(_dir.c_str())))
    {
        perror("opendir");
        return;
    }

    ext_len = strlen(ext);

    if (ext_len == 0)
    {
        while ((dirp = readdir(dp)) != nullptr)
        {
            if (dirp->d_type == DT_DIR)
                result.push_back(dirp->d_name);
        }

    }
    else
    {
        while ((dirp = readdir(dp)) != nullptr)
        {
            if (strcmp(dirp->d_name, ".") == 0 || strcmp(dirp->d_name, "..") == 0)
                continue;

            if (dirp->d_type == DT_DIR)
            {
                temp = _dir + "/" + std::string(dirp->d_name);
                Listdir(temp, ext, result);
            }

            if (strcmp(ext, ".*") == 0)
            {
                //result.push_back(_dir + "/" + std::string(dirp->d_name));
                result.push_back(dirp->d_name);
            }
            else
            {
                len = strlen(dirp->d_name);
                if (strcasecmp(dirp->d_name + (len - ext_len), ext) == 0)
                    result.push_back(_dir + "/" + dirp->d_name);
                else
                    continue;
            }
        }
    }

    closedir(dp);
}
#endif/*Linux*/
#if defined(_WIN64) || defined(_WIN32)
void Listdir(std::string _dir, const char* ext, std::vector<string>& result)
{
    _finddata_t file_info = { 0 };
    intptr_t handel = 0;
    string currentPath = "";
    string temppath = "";
    char _ext[_MAX_EXT];//后缀名

    if (_dir.empty())
    {
        printf("ERROR: Invalid argument _dir (null).\n");
        return;
    }
    if (_access(_dir.c_str(), 0) == -1)//用于判断目录是否存在。如果_access不可用，尝试用access代替
    {
        printf("ERROR: Input directory %s does not exsist\n", _dir.c_str());
        return;
    }

    currentPath = _dir + "/*";
    handel = _findfirst(currentPath.c_str(), &file_info);
    if (-1 == handel)
    {
        printf("ERROR: Maybe the directory %s is empty\n", currentPath.c_str());
        return;
    }

    if (ext == NULL)
    {
        do
        {
            if (file_info.attrib & _A_SUBDIR)
            {
                result.push_back(file_info.name);
            }
        } while (!_findnext(handel, &file_info));
        _findclose(handel);
    }
    else
    {
        do
        {
            if (strcmp(file_info.name, ".") == 0 || strcmp(file_info.name, "..") == 0)
                continue;
            if (file_info.attrib & _A_SUBDIR)
            {
                temppath = _dir + "/" + file_info.name;
                Listdir(temppath, ext, result);
                continue;
            }

            _splitpath(file_info.name, NULL, NULL, NULL, _ext);
            if (strcmp(ext, _ext) != 0)
                continue;
            else
                result.push_back(_dir + "/" + file_info.name);

        } while (!_findnext(handel, &file_info));
        _findclose(handel);
    }
}
#endif/*Windows*/