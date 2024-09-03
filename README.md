# cuda_FNN

The lastest update is under branch 20240524

# Introduction
This project is still under construction, but the forward and the BP process for fully-connected layer (or called Dense layer) have finished. It means the network can work now.
For the convolution neural network version, please move to this project

A C++/CUDA based neural network coding practice, using cublas library.
This program may not work on Linux due to a strange error in curand library.

Tested on Windows 10 with Visual Studio 2019 (VS 2019), CUDA v11.7, MNIST dataset. Due to unkonwn reasons, the program may failed under Release mode. Unfortunately, I can't reproduce this error under Debug mode.
Please run my code under Debug mode if you want to have a try.

# Usage
You can build up a neural network following these steps:
1) declear an object for Net class: Net net;
2) declear some objects for layers, for example: Dense d, df; Flatten flatten;
3) for the first layer of the neural work, you need to fill a corresponding layer information structor. For example, the flatten is your first layer in the neural network, then you should do:
   FlattenInfo fi(...);//fill content in the bracket, there should be instruction shown if you're using VS 2019 or later version
   net.add(&flatten, &fi);//call add() function in Net class
4) add other layers using add() function: net.add(&d, 128, 1.0, "relu"); net.add(df, 10, 1.0, "relu");
5) call net.Forward() for forward process, and call net.Backward() immeidately for training.
