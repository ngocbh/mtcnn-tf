# mtcnn-tf

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