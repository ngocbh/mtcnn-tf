#ifndef __TENSORFLOW_MTCNN_HPP__
#define __TENSORFLOW_MTCNN_HPP__

#include "utils_mtcnn.hpp"
#include "mtcnn.hpp"

void detect_face(TF_Session* sess, TF_Graph * graph, cv::Mat& img, std::vector<face_box>& face_list);

TF_Session * load_graph(const char * frozen_fname, TF_Graph** p_graph);

void successfull(int x);


#endif
