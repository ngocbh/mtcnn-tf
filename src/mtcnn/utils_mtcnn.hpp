#ifndef __UTILS_MTCNN_H__
#define __UTILS_MTCNN_H__

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <opencv2/opencv.hpp>

#include "tensorflow/c/c_api.h"

/* 
   for debug purpose, to save a image or float vector to file.
   the image should be in cv::Mat.
   To avoid OpenCV header file dependency, use void * instead of cv::Mat *
*/
void save_float(const char * name, const float * data, int size);

void save_img(const char * name, void * p_img);

void image_write(const char *name, void * p_img);

#endif
