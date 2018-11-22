#include <iostream>

#include "utils/argument_parser.hpp"
#include "tensorflow/c/c_api.h"
#include "mtcnn/tensorflow_mtcnn.hpp"
#include "mtcnn/mtcnn.hpp"

#define CAMID 0
#define QUIT_KEY 'q'
#define DISP_WINNANE "camera"

using namespace std;

int main(int argc,char* argv[])
{
	parseArgument(argc,argv);
	

	cv::VideoCapture camera;

	camera.open(CAMID);


	if(!camera.isOpened())
	{
		std::cerr<<"failed to open camera"<<std::endl;
		return 1;
	}

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

	double ftick, etick, ticksPerUs;
	ticksPerUs = cv::getTickFrequency() / 1000000;

	cv::Mat frame;
	cv::namedWindow(DISP_WINNANE, cv::WINDOW_AUTOSIZE);

	std::vector<face_box> face_info;
	

	do
	{

		if ( camera.read(frame) ) {

			cv::flip(frame,frame,1);

			ftick = cv::getCPUTickCount();

			detect_face(sess,graph,frame,face_info);

			etick = cv::getCPUTickCount();


			for(unsigned int i=0;i<face_info.size();i++)
			{
				face_box& box=face_info[i];

				/*draw box */

				cv::rectangle(frame, cv::Point(box.x0, box.y0), cv::Point(box.x1, box.y1), cv::Scalar(0, 255, 0), 1);

				/* draw landmark */

				for(int l=0;l<5;l++)
				{
					cv::circle(frame,cv::Point(box.landmark.x[l],box.landmark.y[l]),1,cv::Scalar(0, 0, 255),2);

				}
			}

			std::cout<<"total detected: "<<face_info.size()<<" faces. used "<<(etick - ftick)/ticksPerUs<<" us"<<std::endl;

			cv::imshow(DISP_WINNANE, frame);
			camera.set(CV_CAP_PROP_FPS, 25);
			
	        face_info.clear();
	    }

	} while (QUIT_KEY != cv::waitKey(1));
    // } while (true);

	TF_Status* s = TF_NewStatus();

	TF_CloseSession(sess,s);
	TF_DeleteSession(sess,s);
	TF_DeleteGraph(graph);
	TF_DeleteStatus(s);

	successfull(0);
	return 0;
}   