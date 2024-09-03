# cuda_FNN

# solved problem
This network CANNOT get results better than 11.35%. This is because the network will recognize samples as number one (the second category in MNIST dataset).

It seems that there are something wrong with the dataset I used. I changed another MNIST data source, and everything went well. For a 30 epochs experiment, the accuracy can reach around 90% on the test set. Although not good enough, it finally works.

# Introduction
The forward and the BP process for fully-connected layer (or called Dense layer) have finished.

For the convolution neural network version, please move to this project https://github.com/Purin100/CUDA_CNN

A C++/CUDA based neural network coding practice, using cublas library.
This program may not work on Linux due to a strange error in curand library.

Tested on Windows 10 with Visual Studio 2019 (VS 2019), CUDA v11.7, MNIST dataset. Due to unkonwn reasons, the program will failed under Release mode. Unfortunately, I can't reproduce this error under Debug mode.
Please run my code under Debug mode if you want to have a try. You need OpenCV to load images.

# Usage
You can build up a neural network following these steps:
1) declear an object for Net class: Net net;
2) declear some objects for layers, for example: Dense d, df; Flatten flatten;
3) for the first layer of the neural work, you need to fill a corresponding layer information structor. For example, the flatten is your first layer in the neural network, then you should do:
   FlattenInfo fi(...);//fill content in the bracket, there should be instruction shown if you're using VS 2019 or later version
   net.add(&flatten, &fi);//call add() function in Net class
4) add other layers using add() function: net.add(&d, 128, 1.0, "relu"); net.add(df, 10, 1.0, "relu");
5) call net.Forward() for forward process, and call net.Backward() immeidately for training.
