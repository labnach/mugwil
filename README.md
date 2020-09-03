# Mugwil

Mugwil is a Music Generator based on LSTMs.

The model is able to generate music from an input in ABC format. This music notation uses only ASCII characters, so it can be easily fed into RNNs. You can find details and examples of ABC notation here: http://abcnotation.com/.

The dataset used to train the model is the [ABC version of the Nottingham Music Database](http://abc.sourceforge.net/NMD/).

This repository includes an implementation of the model using Keras (TensorFlow 1.15 as backend), utilities to download and process the dataset, and Jupyter notebooks to explore the input and results.


## Requirements

* [Docker](https://docs.docker.com/get-docker/)
* [nvidia-docker2](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) (optional, to train using GPU)


## Setup

A shell script is provided to simplify the setup process. Run:
```
MODE=cpu ./dev/tools.sh build
```

Then to launch the Jupyter environment:
```
MODE=cpu ./dev/tools.sh run
```

If you want to use a NVIDIA GPU for training, remove `MODE=cpu` from the command. Remember to install the requirements for that: NVIDIA drivers and nvidia-docker2 package.
