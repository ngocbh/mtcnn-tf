#include <iostream>
#include "utils/argument_parser.h"
#include "tensorflow/c/c_api.h"

using namespace std;

int main(int argc,char* argv[])
{
	parseArgument(argc,argv);
	cout << INPUT_IMAGE << endl;
}