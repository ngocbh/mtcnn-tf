#include "utils_mtcnn.hpp"
#include "mtcnn.hpp"

static int load_file(const std::string & fname, std::vector<char>& buf)
{
	std::ifstream fs(fname, std::ios::binary | std::ios::in);

	if(!fs.good())
	{
		std::cerr<<fname<<" does not exist"<<std::endl;
		return -1;
	}


	fs.seekg(0, std::ios::end);
	int fsize=fs.tellg();

	fs.seekg(0, std::ios::beg);
	buf.resize(fsize);
	fs.read(buf.data(),fsize);

	fs.close();

	return 0;

}


//--------------------------------------------------------------
// Load frozen_model to graph
//--------------------------------------------------------------
TF_Session * load_graph(const char * frozen_fname, TF_Graph ** p_graph)
{
	//--------------------------------------------------------------
	// TF_Status : Denotes success or failure of a call in Tensorflow.
	//--------------------------------------------------------------
	TF_Status* s = TF_NewStatus();

	TF_Graph* graph = TF_NewGraph();

	std::vector<char> model_buf;

	load_file(frozen_fname,model_buf);

	TF_Buffer graph_def = {model_buf.data(), model_buf.size(), nullptr};

	TF_ImportGraphDefOptions* import_opts = TF_NewImportGraphDefOptions();
	TF_ImportGraphDefOptionsSetPrefix(import_opts, "");
	TF_GraphImportGraphDef(graph, &graph_def, import_opts, s);

	if(TF_GetCode(s) != TF_OK)
	{
		printf("load graph failed!\n Error: %s\n",TF_Message(s));
		return nullptr;
	}

	TF_SessionOptions* sess_opts = TF_NewSessionOptions();
	TF_Session* session = TF_NewSession(graph, sess_opts, s);
	assert(TF_GetCode(s) == TF_OK);


	TF_DeleteStatus(s);

	*p_graph=graph;

	return session;
}

//--------------------------------------------------------------
// 
//--------------------------------------------------------------
void generate_bounding_box_tf(const float * confidence_data, int confidence_size,
		const float * reg_data, float scale, float threshold, 
		int feature_h, int feature_w, std::vector<face_box>&  output, bool transposed)
{

	int stride = 2;
	int cellSize = 12;

	int img_h = feature_h;
	int img_w = feature_w;

	// std::cout << img_h << " " << img_w << " " << scale << '\n';

	for(int y=0;y<img_h;y++)
		for(int x=0;x<img_w;x++)
		{
			int line_size=img_w*2;

			float score=confidence_data[line_size*y+2*x+1];

			if(score>= threshold)
			{
				// std::cout << x << " " << y << " " << score << std::endl;
				// why x*stride + 1 ???
				// it stride may be in pool1 layer
				// it is supplement when feature map go through pool1
				// the other layer like conv, relu have stride = 1. so it's not need to supplement
				float top_x = (int)((x*stride + 1) / scale);
				float top_y = (int)((y*stride + 1) / scale);
				float bottom_x = (int)((x*stride + cellSize) / scale);
				float bottom_y = (int)((y*stride + cellSize) / scale);

				face_box box;

				box.x0 = top_x;
				box.y0 = top_y;
				box.x1 = bottom_x;
				box.y1 = bottom_y;

				box.score=score;

				int c_offset=(img_w*4)*y+4*x;
				// why? what dose reg_data contain?
				// it contain bounding box regression parameter to estimate ground truth box of this bounding box
				// Gx = Pw * dx(Phi(P)) + Px
				// Gy = Ph * dy(Phi(P)) + Py 
				// Gw = Pw * exp(dw(Phi(P)))
				// Gh = Ph * exp(dh(Phi(P)))
				// P is this bounding box proprosal , Phi(P) is feature map of bounding box proprosal
				// for each proposal P it generate 4 parameter ( dx(P), dy(P), dw(P), dh(P) ) to calculate new ground truth box
				// bounding box regression base on heatmap ( map of feature ). it's difference with geometric bounding box regression
				if(transposed)
				{

					box.regress[1]=reg_data[c_offset];
					box.regress[0]=reg_data[c_offset+1]; 
					box.regress[3]=reg_data[c_offset+2];
					box.regress[2]=reg_data[c_offset+3];
				}
				else {

					box.regress[0]=reg_data[c_offset];
					box.regress[1]=reg_data[c_offset+1]; 
					box.regress[2]=reg_data[c_offset+2];
					box.regress[3]=reg_data[c_offset+3];
				}

				output.push_back(box);
			}

		}
}

/* To make tensor release happy...*/
static void dummy_deallocator(void* data, size_t len, void* arg)
{
}

void run_PNet(TF_Session * sess, TF_Graph * graph, cv::Mat& img, scale_window& win, std::vector<face_box>& box_list)
{
	cv::Mat  resized;
	int scale_h=win.h;
	int scale_w=win.w;
	float scale=win.scale;
	float pnet_threshold=0.6;

	cv::resize(img, resized, cv::Size(scale_w, scale_h),0,0);

	/* tensorflow related*/

	TF_Status * s= TF_NewStatus();

	// std::cout << scale_w << " " << scale_h << std::endl;
	// set input for pnet
	std::vector<TF_Output> input_names;
	std::vector<TF_Tensor*> input_values;

	TF_Operation* input_name=TF_GraphOperationByName(graph, "pnet/input");

	input_names.push_back({input_name, 0});

	const int64_t dim[4] = {1,scale_h,scale_w,3};

	TF_Tensor* input_tensor = TF_NewTensor(TF_FLOAT,dim,4,resized.ptr(),sizeof(float)*scale_w*scale_h*3,dummy_deallocator,nullptr);

	input_values.push_back(input_tensor);


	//set output for pnet
	std::vector<TF_Output> output_names;

	TF_Operation* output_name = TF_GraphOperationByName(graph,"pnet/conv4-2/BiasAdd");
	output_names.push_back({output_name,0});

	output_name = TF_GraphOperationByName(graph,"pnet/prob1");
	output_names.push_back({output_name,0});

	std::vector<TF_Tensor*> output_values(output_names.size(), nullptr);

	

	//run pnet
	TF_SessionRun(sess,nullptr,input_names.data(),input_values.data(),input_names.size(),
			output_names.data(),output_values.data(),output_names.size(),
			nullptr,0,nullptr,s);


	/*retrieval the forward results*/

	// conf_data : output from pnet/prob1 ( from fully connected layer )
	const float * conf_data=(const float *)TF_TensorData(output_values[1]);
	// reg_data : output from pnet/conv4-2/BiasAdd 
	const float * reg_data=(const float *)TF_TensorData(output_values[0]);


	int feature_h=TF_Dim(output_values[0],1);
	int feature_w=TF_Dim(output_values[0],2);

	int conf_size=feature_h*feature_w*2;

	std::vector<face_box> candidate_boxes;

	generate_bounding_box_tf(conf_data,conf_size,reg_data, 
			scale,pnet_threshold,feature_h,feature_w,candidate_boxes,true);

	// for (int i = 0; i < candidate_boxes.size(); i++)
	// 	std::cout << candidate_boxes[i] << '\n';
	// std::cout << std::endl;


	nms_boxes(candidate_boxes, 0.5, NMS_UNION,box_list);

	TF_DeleteStatus(s);
	TF_DeleteTensor(output_values[0]);
	TF_DeleteTensor(output_values[1]);
	TF_DeleteTensor(input_tensor);
}

void run_RNet(TF_Session * sess, TF_Graph * graph, cv::Mat& img, std::vector<face_box>& pnet_boxes, std::vector<face_box>& output_boxes)
{

}


void detect_face(TF_Session * sess, TF_Graph * graph, cv::Mat& img, std::vector<face_box>& face_list)
{
	cv::Mat working_img;

	float alpha=0.0078125;
	float mean=127.5;

	//convert type of Mat to 32-bit floating-point numbers
	img.convertTo(working_img, CV_32FC3);
	working_img = img;
	//resize value of mat by scaling
	working_img=(working_img-mean)*alpha;
	//transpose image ??? WHY
	working_img=working_img.t();
	// WHY???
	cv::cvtColor(working_img,working_img, cv::COLOR_BGR2RGB);

	int img_h=working_img.rows;
	int img_w=working_img.cols;

	int min_size=40;
	float factor=0.709;

	std::vector<scale_window> win_list;

	std::vector<face_box> total_pnet_boxes;
	std::vector<face_box> total_rnet_boxes;
	std::vector<face_box> total_onet_boxes;

	calc_scale_pyramid(img_h,img_w,min_size,factor,win_list);

	for(unsigned int i=0;i<win_list.size();i++)
	{
		std::vector<face_box>boxes;

		run_PNet(sess,graph,working_img,win_list[i],boxes);
		
		total_pnet_boxes.insert(total_pnet_boxes.end(),boxes.begin(),boxes.end());
	}

	std::vector<face_box> pnet_boxes;
	process_boxes(total_pnet_boxes,img_h,img_w,pnet_boxes);

	// RNet
	std::vector<face_box> rnet_boxes;

	run_RNet(sess, graph,working_img, pnet_boxes,total_rnet_boxes);

	process_boxes(total_rnet_boxes,img_h,img_w,rnet_boxes);
}

void successfull(int x)
{ 
	std::cout << "RUNNING SUCCESSFULL WITH FLAG : " <<  x << std::endl;
}