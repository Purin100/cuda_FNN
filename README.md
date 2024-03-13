# cuda_FNN
This project is still under construction, but the forward and the BP process for fully-connected layer (or called Dense layer) have finished. It means the network can work now.
I will add convolutional layer in the future.

A C++/CUDA based neural network coding practice, using cublas library.
This program may not work on Linux due to a strange error in curand library.

mnist-testsamples.zip and mnist-trainsamples.7z are data from MNIST dataset, I saved them in .txt format. Each txt file is a sample. Labels are saved in train_label.txt and test_label.txt.

Tested on Windows 10 with MSVC 2019, CUDA v11.7, MNIST dataset. Due to unkonwn reasons, the program will failed under Release mode. Unfortunately, I can't reproduce this error under Debug mode.
Please run my code under Debug mode if you want to have a try. Besides, it takes very long time to train the neural network (more than 500 epochs before I can reach 50% accuracy).
