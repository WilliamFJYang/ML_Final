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
  IplImage *SLIC_image = cvCloneImage(&(IplImage)image);
  IplImage *SLIC_lab_image = cvCloneImage(SLIC_image);
  cvCvtColor(SLIC_image, SLIC_lab_image, CV_BGR2Lab);

  /* Yield the number of superpixels and weight-factors from the user. */
  int w = SLIC_image->width, h = SLIC_image->height;
  int nr_superpixels = 400;
  int nc = 40;

  double step = sqrt((w * h) / (double)nr_superpixels);

  /* Perform the SLIC superpixel algorithm. */
  Slic slic;
  slic.generate_superpixels(SLIC_lab_image, step, nc);
  slic.create_connectivity(SLIC_lab_image);

  /* Display the contours and show the result. */
  slic.display_contours(SLIC_image, CV_RGB(255, 0, 0));
  cvShowImage("result", SLIC_image);
  return 0;
}
