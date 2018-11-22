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

	unsigned long start_time=get_cur_time();

	detect_face(sess,graph,img,face_list);

	unsigned long end_time=get_cur_time();

	
	for(unsigned int i=0;i<face_list.size();i++)
	{
		face_box& box=face_list[i];

		printf("face %d: x0,y0 %2.5f %2.5f  x1,y1 %2.5f  %2.5f conf: %2.5f\n",i,
				box.x0,box.y0,box.x1,box.y1, box.score);
		printf("landmark: ");

		for(unsigned int j=0;j<5;j++)
			printf(" (%2.5f %2.5f)",box.landmark.x[j], box.landmark.y[j]);

		printf("\n");


		if(SAVE_CHOP)
		{

			cv::Mat corp_img=img(cv::Range(box.y0,box.y1),
					cv::Range(box.x0,box.x1));

			char title[128];

			sprintf(title,"id%d.jpg",i);

			cv::imwrite(title,corp_img);
		}

		/*draw box */

		cv::rectangle(img, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 1);


		/* draw landmark */

		for(int l=0;l<5;l++)
		{
			cv::circle(img,cv::Point(box.landmark.x[l],box.landmark.y[l]),1,cv::Scalar(0, 255, 0),1);
		}
	}

	cv::imwrite(OUTPUT_IMAGE,img);

	std::cout<<"total detected: "<<face_list.size()<<" faces. used "<<(end_time-start_time)/1000000.0<<" seconds"<<std::endl;
	std::cout<<"boxed faces are in file: "<<OUTPUT_IMAGE<<std::endl;

	TF_Status* s = TF_NewStatus();

	TF_CloseSession(sess,s);
	TF_DeleteSession(sess,s);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(s);

	successfull(0);
}   