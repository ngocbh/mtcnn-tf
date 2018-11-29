# MTCNN TensorFlow C++ Implementation

## Build

#### Install opencv
* For Unix user
```
brew install opencv
brew install pkg-config
```

#### Tensorflow build
* Flow this [!tutorial](https://www.tensorflow.org/install/source) to build tensorflow c api. A brief: 
```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
bazel build --config=opt //tensorflow/tools/lib_package:libtensorflow
```
* Copy file bazel-bin/tensorflow/tools/lib_package/libtensorflow.tar.gz to mtcnn-tf directory and unzip it
```
mkdir -p libs
cd libs
(Copy libtensorflow.tar.gz to this directory)
tar -xvzf libtensorflow.tar.gz
```

#### Compile by make
```
make -j4
```

## Run

#### Test detect image
```
./bin/detect_face --input test.jpg --output test-detected.jpg
```

#### Test camera
```
./bin/camera_detect
```

### tensorflow docs

* TF_Session : run a tensor to export output ( the real value )
* TF_Graph : represent dataflow graph
* TF_Status : Denotes success or failure of a call in Tensorflow.

### opencv docs

* cv::Mat : image stored
* frame.size() shape of Mat ( cout << frame.size() << endl; )
* cv::imread(fname [,flag]) : read image from file and Flags specifying the color type of a loaded image
* cv::convertTo(img,CV_32FC3) : convert type of Mat to 32-bit floating-point numbers
* cv::cvtColor(working_img,working_img, cv::COLOR_BGR2RGB); 