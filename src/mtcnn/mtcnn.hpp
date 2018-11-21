#ifndef __MTCNN_HPP__
#define __MTCNN_HPP__

#include "utils_mtcnn.hpp"

#define NMS_UNION 1
#define NMS_MIN  2
//--------------------------------------------------------------
// 5 point landmark 
// 0 : | 1 : | 2 : | 3 : | 4 : |
//--------------------------------------------------------------
struct face_landmark
{
	float x[5];
	float y[5];
};

//--------------------------------------------------------------
// face box detected
//--------------------------------------------------------------
struct face_box
{
	float x0;
	float y0;
	float x1;
	float y1;

	/* confidence score */
	float score;

	/*regression scale */

	float regress[4];

	/* padding stuff*/
	float px0;
	float py0;
	float px1;
	float py1;

	face_landmark landmark;  
};

std::ostream& operator<<(std::ostream& os,const face_box a);

struct scale_window
{
	int h;
	int w;
	float scale;
};

class mtcnn {
	public:
		mtcnn(void){
			min_size_=40;
			pnet_threshold_=0.6;
			rnet_threshold_=0.7;
			onet_threshold_=0.9;
			factor_=0.709;

		}

		void set_threshold(float p, float r, float o)
		{
			pnet_threshold_=p;
			rnet_threshold_=r;
			onet_threshold_=o;
		}

		void set_factor_min_size(float factor, float min_size)
		{
			factor_=factor;
			min_size_=min_size;   
		}
		

		virtual int load_model(const std::string& model_dir)=0;
		virtual void detect(cv::Mat& img, std::vector<face_box>& face_list)=0;
		virtual ~mtcnn(void){};

	protected:

		int min_size_;
		float pnet_threshold_;
		float rnet_threshold_;
		float onet_threshold_;
		float factor_;	 
};  

void  calc_scale_pyramid(int height, int width, int min_size, float factor,std::vector<scale_window>& list);

void nms_boxes(std::vector<face_box>& input, float threshold, int type, std::vector<face_box>&output);

void regress_boxes(std::vector<face_box>& rects);

void process_boxes(std::vector<face_box>& input, int img_h, int img_w, std::vector<face_box>& rects);

#endif