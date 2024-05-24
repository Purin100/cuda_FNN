# cuda_FNN

This network CANNOT get results better than 11.35%. This is because the network will recognize samples as number one (the second category in MNIST dataset). I am trying to fix this problem, but I cannot figure out what causes this strange problem now. 

This project is still under construction, but the forward and the BP process for fully-connected layer (or called Dense layer) have finished. It means the network can work now.
I will add convolutional layer in the future.

A C++/CUDA based neural network coding practice, using cublas library.
This program may not work on Linux due to a strange error in curand library.

mnist-testsamples.zip and mnist-trainsamples.7z are data from MNIST dataset, I saved them in .txt format. Each txt file is a sample. Labels are saved in train_label.txt and test_label.txt.

Tested on Windows 10 with Visual Studio 2019 (VS 2019), CUDA v11.7, MNIST dataset. Due to unkonwn reasons, the program will failed under Release mode. Unfortunately, I can't reproduce this error under Debug mode.
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
