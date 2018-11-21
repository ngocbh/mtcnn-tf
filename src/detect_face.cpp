#include <iostream>

#include "utils/argument_parser.hpp"
#include "tensorflow/c/c_api.h"
#include "mtcnn/tensorflow_mtcnn.hpp"
#include "mtcnn/mtcnn.hpp"

using namespace std;

int main(int argc,char* argv[])
{
	parseArgument(argc,argv);
	
	//--------------------------------------------------------------
	// TF_Session : run a tensor to export output ( the real value )
	//--------------------------------------------------------------
	TF_Session * sess;
	//--------------------------------------------------------------
	// TF_Graph : represent dataflow graph
	//--------------------------------------------------------------
	TF_Graph * graph;	

	sess = load_graph(PRETRAINED_MODEL.c_str(),&graph);

	if(sess==nullptr)
		return 1;

	//Load image
	cv::Mat img = cv::imread(INPUT_IMAGE);

	if(!img.data)
	{
		cerr<<"failed to read image file: "<< INPUT_IMAGE << endl;
		return 1; 
	}

	vector<face_box> face_list;

	detect_face(sess,graph,img,face_list);

	cv::imwrite(OUTPUT_IMAGE,img);

	successfull(0);
}   