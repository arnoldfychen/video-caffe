# video-caffe
C3D version Caffe, with some specific changes for Jetson Xavier NX or Nano

# WTR cuDNN8
The original version video-caffe can only work with cuDNN7.x, but since CUDA11.x, only cudnn8 is supported, so I made some changes in code to make video-caffe can work with CUDA11.x + cuDNN8.x.

I have made code changes in fo cudnn8:

       cmake/Cuda.cmake
       src/caffe/layers/cudnn_ndconv_layer.cu
       src/caffe/layers/cudnn_conv_layer.cpp
       src/caffe/layers/cudnn_deconv_layer.cpp

If you do want to use video-caffe with cudnn8, you need to make these changes in the above files:

    1) Open the file “cmake/Cuda.cmake”.  replace "cudnn.h" with "cudnn_version.h" by commenting/uncommenting the lines where they are.
    2) In cudnn_ndconv_layer.cu, cudnn_conv_layer.cpp and cudnn_deconv_layer.cpp  change all "#if 0 // CUDNN_VERSION_MIN(8, 0, 0)" to "if CUDNN_VERSION_MIN(8, 0, 0)".

# As cuDNN8.x is supported by the above changes, the following requirements of installing cuda10.2 and libcudnn7_7.x + libcudnn7-dev_7.x are not necessary now, you can use video-caffe with cuda11.x + cudnn8.x 
# Requirements \[Deprecated\]
Make sure that cuda10.2 and libcudnn7_7.x + libcudnn7-dev_7.x and opencv3/opencv4 are installed on your NX or Nano, if not, you can install cuda10.2 and opencv by NVIDIA's sdkmanager(Jet Pack4.4), and install libcudnn7_7.x + libcudnn7-dev_7.x by deb packages. As video-caffe uses cudnn API whose version is not higher than 7.x, but sdkmanager install cudnn 8.x on Xavier NX or Nano by default, so you have to install libcudnn7_7.x + libcudnn7-dev_7.x by manual. 

install cuda10.2:

    1)sudo dpkg -i cuda-repo-l4t-10-2-local-10.2.89_1.0-1_arm64.deb
    2)sudo apt-get update
    3)sudo apt-get install cuda-toolkit-10-2

install cudnn7.6.3 for cuda10.0 (it actually also works for cuda10.2):

    1)sudo dpkg -i libcudnn7_7.6.3.28-1+cuda10.0_arm64.deb
    2)sudo dpkg -i libcudnn7-dev_7.6.3.28-1+cuda10.0_arm64.deb 

If you installed your NX or Nano with sdkmanager, cuda10.2 should be there and you don't need install it by manual, but you may have to install cudnn7.6.3 by manual as cudnn8.x is installed by sdkmanager by default. Please note you don't need remove cudnn8.x before you install cudnn7.6.3 as they can exist together.


# How to use

1.For a new Xavier NX or Nano, login with root and install the following tools and packages required by caffe:
  
    1)Remove old version cmake and get the latest source CMake code, and compile and install it on NX:    
       apt-get remove cmake
       wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3.tar.gz
       tar xf cmake-3.17.3.tar.gz
       cd cmake
       ./bootstrap
       make
       make install
  
    2)Make Pythonr3.6 instead of Python2.7 as the default Python with Utbuntu's update-alternatives and upgrade pip and setuptools
       #Remember to make Python3.6 as the default Python:
       update-alternatives --install /usr/bin/python python /usr/bin/python2.7 10
       update-alternatives --install /usr/bin/python python /usr/local/bin/python3.6 20

       #then:
       pip install -U pip
       pip install -U setuptools
       pip install python-dateutil   #Optional
    
    3)Install other tools and packages:
       apt-get install git cmake-qt-gui 
       apt-get install libprotobuf-dev protobuf-compiler libleveldb-dev libsnappy-dev libhdf5-serial-dev libgflags-dev libgoogle-glog-dev liblmdb-dev libatlas-base-dev gfortran libssl-dev 
       apt-get install --no-install-recommends libboost-all-dev
       apt-get install python-dev python-numpy
  
  
2. Download the source code and build out this specific C3D caffe:

       git clone https://github.com/BrightDotAi/video-caffe
       cd video-caffe
       mkdir build && cd build
       make all -j6
       make install
  
3.Copy the .so files under build/lib/ to the lib directory of our DeepStream application, e.g., /home/jzyq/bright/ext/lib/, 
  and copy the directory build/inclue to application's include path or just add build/include into application's include path.

4.For applying video-caffe model to applications, please also copy c3d_ucf101_deploy.prototxt and the model weights file that you train out, such as c3d_ucf101_iter_4999.caffemodel into your application to application’s directory, and set the full path of them as the parameters for Classifier.Load(string model_file,string trained_file). 

Please note do the following changes if you face errors like "error: ‘CV_LOAD_IMAGE_COLOR’ was not declared in this scope"(this implies your opencv version is too new, e.g. 4.1, as CV_LOAD_IMAGE_COLOR is an opencv 3.x definition):

   1) In src/caffe/util/io.cpp and src/caffe/layers/window_data_layer.cpp, change CV_LOAD_IMAGE_COLOR to cv::IMREAD_COLOR, and change CV_LOAD_IMAGE_GRAYSCALE  to cv::IMREAD_GRAYSCALE .

   2) In src/caffe/util/io.cpp, change CV_CAP_PROP_FRAME_COUNT to cv::CAP_PROP_FRAME_COUNT, and change CV_CAP_PROP_POS_FRAMES  to cv::CAP_PROP_POS_FRAMES .


# Original README:
# Video-Caffe: Caffe with C3D implementation and video reader

[![Build Status](https://travis-ci.org/chuckcho/video-caffe.svg?branch=master)](https://travis-ci.org/chuckcho/video-caffe)

This is 3D convolution (C3D) and video reader implementation in the latest Caffe (Dec 2016). The original [Facebook C3D implementation](https://github.com/facebook/C3D/) is branched out from Caffe on July 17, 2014 with git commit [b80fc86](https://github.com/BVLC/caffe/tree/b80fc862952ba4e068cf74acc0823785ce1cc0e9), and has not been rebased with the original Caffe, hence missing out quite a few new features in the recent Caffe. I therefore pulled in C3D concept and an accompanying video reader and applied to the latest Caffe, and will try to rebase this repo with the upstream whenever there is a new important feature. This repo is rebased on [99bd997](https://github.com/BVLC/caffe/commit/99bd99795dcdf0b1d3086a8d67ab1782a8a08383), on Aug`21 2018.
Please reach [me](https://github.com/chuckcho) for any feedback or question.

Check out the [original Caffe readme](README-original.md) for Caffe-specific information.

## Branches

[`refactor` branch](https://github.com/chuckcho/video-caffe/tree/refactor) is a recent re-work, based on the [original Caffe](https://github.com/BVLC/caffe) and [Nd convolution and pooling with cuDNN PR](https://github.com/BVLC/caffe/pull/3983). This is a cleaner, less-hacky implementation of 3D convolution/pooling than the `master` branch, and is supposed to more stable than the `master` branch. So, feel free to try this branch. One missing feature in the `refactor` branch (yet) is the python wrapper.

## Requirements

In addition to [prerequisites for Caffe](http://caffe.berkeleyvision.org/installation.html#prerequisites), video-caffe depends on cuDNN. It is known to work with CuDNN verson 4 and 5, but it may need some efforts to build with v3.

* If you use "make" to build make sure `Makefile.config` point to the right paths for CUDA and CuDNN.
* If you use "cmake" to build, double-check `CUDNN_INCLUDE` and `CUDNN_LIBRARY` are correct. If not, you may want something like `cmake -DCUDNN_INCLUDE="/your/path/to/include" -DCUDNN_LIBRARY="/your/path/to/lib" ${video-caffe-root}`.

## Building video-caffe

Key steps to build video-caffe are:

1. `git clone git@github.com:chuckcho/video-caffe.git`
2. `cd video-caffe`
3. `mkdir build && cd build`
4. `cmake ..`
5. Make sure CUDA and CuDNN are detected and their paths are correct.
6. `make all -j8`
7. `make install`
8. (optional) `make runtest`

## Usage

Look at [`${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_train_test.prototxt`](examples/c3d_ucf101/c3d_ucf101_train_test.prototxt) for how 3D convolution and pooling are used. In a nutshell, use `NdConvolution` or `NdPooling` layer with `{kernel,stride,pad}_shape` that specifies 3D shapes in (L x H x W) where `L` is the temporal length (usually 16).
```
...
# ----- video/label input -----
layer {
  name: "data"
  type: "VideoData"
  top: "data"
  top: "label"
  video_data_param {
    source: "examples/c3d_ucf101/c3d_ucf101_train_split1.txt"
    batch_size: 50
    new_height: 128
    new_width: 171
    new_length: 16
    shuffle: true
  }
  include {
    phase: TRAIN
  }
  transform_param {
    crop_size: 112
    mirror: true
    mean_value: 90
    mean_value: 98
    mean_value: 102
  }
}
...
# ----- 1st group -----
layer {
  name: "conv1a"
  type: "NdConvolution"
  bottom: "data"
  top: "conv1a"
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    kernel_shape { dim: 3 dim: 3 dim: 3 }
    stride_shape { dim: 1 dim: 1 dim: 1 }
    pad_shape    { dim: 1 dim: 1 dim: 1 }
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
  }
}
...
layer {
  name: "pool1"
  type: "NdPooling"
  bottom: "conv1a"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_shape { dim: 1 dim: 2 dim: 2 }
    stride_shape { dim: 1 dim: 2 dim: 2 }
  }
}
...
```

## UCF-101 training demo

Scripts and training files for C3D training on UCF-101 are located in [examples/c3d_ucf101/](examples/c3d_ucf101/).
Steps to train C3D on UCF-101:

1. Download UCF-101 dataset from [UCF-101 website](http://crcv.ucf.edu/data/UCF101.php).
2. Unzip the dataset: e.g. `unrar x UCF101.rar`
3. (Optional) video reader works more stably with extracted frames than directly with video files. Extract frames from UCF-101 videos by revising and running a helper script, [`${video-caffe-root}/examples/c3d_ucf101/extract_UCF-101_frames.sh`](examples/c3d_ucf101/extract_UCF-101_frames.sh).
4. Change `${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_{train,test}_split1.txt` to correctly point to UCF-101 videos or directories that contain extracted frames.
5. Modify [`${video-caffe-root}/examples/c3d_ucf101/c3d_ucf101_train_test.prototxt`](examples/c3d_ucf101/c3d_ucf101_train_test.prototxt) to your taste or HW specification. Especially `batch_size` may need to be adjusted for the GPU memory.
6. Run training script: e.g. `cd ${video-caffe-root} && examples/c3d_ucf101/train_ucf101.sh` (optionally use `--gpu` to use multiple GPU's)
7. (Optional) Occasionally run [`${video-caffe-root}/tools/extra/plot_training_loss.sh`](tools/extra/plot_training_loss.sh) to get training loss / validation accuracy (top1/5) plot. It's pretty hacky, so look at the file to meet your need.
8. At 7 epochs of training, clip accuracy should be around 45%.

A typical training will yield the following loss and top-1 accuracy: ![iter-loss-accuracy plot](examples/c3d_ucf101/c3d_ucf101_train_loss_accuracy.png?raw=true "Iteration vs Training loss and top-1 accuracy")

## Pre-trained model

A pre-trained model is available ([downloadable link](https://www.dropbox.com/s/gglm2c67154nltr/c3d_ucf101_iter_20000.caffemodel?dl=0)) for UCF101 (trained from scratch), achieving top-1 accuracy of ~47%.

## To-do
1. Feature extractor script.
2. Python demo script that loads a video and classifies.
3. Convert Sport1M pre-trained model and make it available.

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
