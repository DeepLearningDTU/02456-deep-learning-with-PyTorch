# 02456-deep-learning-with-PyTorch

This repository contains exercises for the DTU course 02456 Deep Learning. The exercises are written in Python programming language and formatted into Jupyter Notebooks.

This repository borrows heavily from previous works, in particular:

* [2015 DTU Summerschool in Deep Learning](https://github.com/DeepLearningDTU/Summerschool_2015/tree/master/day1-NN). A PhD summerschool that was held at DTU in 2015. Exercises both in **numpy** and **Theano**.

* [02456-deep-learning](https://github.com/DeepLearningDTU/02456-deep-learning). Previous version of the course material for this course, but using **TensorFlow** for the exercises.

* [Pytorch Tutorial](https://github.com/munkai/pytorch-tutorial). A remix popular deep learning materials, including material from 02456, collected in one coherent package using **PyTorch**, with a focus on natural language processing (NLP)

* [pytorch/tutorials](https://github.com/pytorch/tutorials). Official tutorials from the PyTorch repo.

## Setup
We will use Docker to manage the software needed for the exercises.
* **NB:** If you have a **NVIDIA GPU** you should follow the GPU instructions below, as a GPU enables you to run the exercises considerably faster.
* **NB:** If you are using **Windows** things are a bit more difficult. We will help you the best we can, but be prepared to do some resaerch on your own.


### CPU (Linux & mac)
First you need to [install Docker](https://docs.docker.com/install/).

Setup Docker by typing this in the command line in the exercise folder (this one)

* ```docker build -t munkai/pytorch:cpu -f Dockerfile.cpu .```

Start Docker container

* ```docker run -it -p 8888:8888 -v `pwd`:/work munkai/pytorch:cpu ./jupyter_run.sh```

Go to your browser, and type in `http://localhost:8888`. It will ask for a password (token) that you can copy from your terminal.

### GPU (Linux & mac)
Setting up your GPU may take a while, and you might need to consult your favorite search engine.
You need Nvidia and nvidia-docker installed for this.

Make sure you have Nvidia's drivers installed for your system.
The folowing instructions will install CUDA and NVIDA drivers on ubuntu 16.04.
Adjust as appropriate.

```
DISTRO=ubuntu
VERSION=1604
ARCH=x86_64
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}${VERSION}/${ARCH}/7fa2af80.pub
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}${VERSION}/${ARCH}/" > /etc/apt/sources.list.d/cuda.list'
sudo apt-get update && sudo apt-get install -y --no-install-recommends cuda-drivers
```

Install nvidia-docker: See https://github.com/NVIDIA/nvidia-docker on how to install `nvidia-docker`.

Setup Docker with GPU by typing this in the command line in the exercise folder (this one)

* ```docker build -t munkai/pytorch:gpu -f Dockerfile.gpu .```

Running docker with a CUDA-enabled machine

* ```nvidia-docker run -it -p 8888:8888 -v `pwd`:/work munkai/pytorch:gpu ./jupyter_run.sh```

Go to your browser, and type in `http://localhost:8888`. It will ask for a password (token) that you can copy from your terminal.

### CPU (Windows)
The following instructions will help you setup Docker on Windows. 

1. Install [install Docker](https://docs.docker.com/install/)
1. Make drive shareable - complete steps 1-3 in [this guide](https://rominirani.com/docker-on-windows-mounting-host-directories-d96f3f056a2c)
1. Setup Docker by typing this in the command line in the exercise folder (this one)
 1. ```docker build -t munkai/pytorch:cpu -f Dockerfile.cpu .```
1. Make sure that `jupyter_run.sh` has Unix style line endings (git has probaly made it Windos style when you downloaded. Text editors like Sublime can change that).
1. Run Docker (change command to match your setup)
 1. ```docker run -v c:/PATH/TO/EXERCISES/02456-deep-learning-with-PyTorch:/work -it --rm -p 8888:8888 munkai/pytorch:cpu ./jupyter_run.sh```
1. Go to your browser, and type in `http://localhost:8888`. It will ask for a password (token) that you can copy from your terminal.

And you are done! 
Once setup is complete you only need to perform the last 2 steps to get up and running.


**Debugging**

If you are having issues we have made a list of problems and solutions to help you.
Plase help us extend this list and help people in the future by letting us know about you issues and the solutions you found.

* Jupyter starts, but you don't see the exercises.
  * The drive isn't shared properly. Take a look at [this guide](https://rominirani.com/docker-on-windows-mounting-host-directories-d96f3f056a2c) again, or [this Stack Overflow question](https://stackoverflow.com/questions/23439126/how-to-mount-a-host-directory-in-a-docker-container).

* ```standard_init_linux.go:190: exec user process caused "no such file or directory"```
  * https://github.com/docker/labs/issues/215


### GPU (Windows)
We haven't tested this, but it should be easy to combine the *GPU (Linux & mac)* and *CPU (Windows)* guides above.
Let us know if this works/doesn't work for you.


## 7. Additional content

If you're interested in some PyTorch codebases check out the following links (reinforcement learning, GANTs, ResNet, etc).

- [Train neural nets to play video games](http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Train a state-of-the-art ResNet network on imagenet](https://github.com/pytorch/examples/tree/master/imagenet)
- [Train a face generator using Generative Adversarial Networks](https://github.com/pytorch/examples/tree/master/dcgan)
- [Train a word-level language model using Recurrent LSTM networks](https://github.com/pytorch/examples/tree/master/word_language_model)
- [More examples](https://github.com/pytorch/examples)
- [More tutorials](https://github.com/pytorch/tutorials)
- [Discuss PyTorch on the Forums](https://discuss.pytorch.org/)
- [Chat with other users on Slack](http://pytorch.slack.com/messages/beginner/)