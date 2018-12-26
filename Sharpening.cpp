#include "stdafx.h"
#include "myHist.h"
#include "slic.h"
#include <iostream>

#include <stdio.h>
#include <string.h>
#include <io.h>
#include <direct.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <float.h>

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
int main() {
	const string read_dir = "../input/";
	const string fileName[] = { "000.bmp" };
  Mat image = imread(read_dir + fileNametemp, 1);
  Mat GX; Sobel(imageBGR, GX, 0, 1, 0, 3, 1, 0, BORDER_DEFAULT);
  Mat GY; Sobel(imageBGR, GY, 0, 0, 1, 3, 1, 0, BORDER_DEFAULT);
  imageBGR = imageBGR - GX * 0.25 - GY * 0.25;
  imshow("銳化", imageBGR);
  imwrite("../_銳化.bmp", imageBGR);.
  return 0;
}
