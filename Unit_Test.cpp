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

const int LBP_58_table[58] = { 0,128,192,224,240,248,252,254,64,96,112,120,124,126,254,32,48,56,60,62,190,254,16,24,28,30,158,222,254,8,12,14,142,206,238,254,4,6,134,198,230,246,254,2,130,194,226,242,250,254,128,192,224,240,248,252,254,255 };

Mat Hist(Mat image) {
	Mat canvasInput_in;
	canvasInput_in.create(Size(512, 512), CV_8UC1);
	canvasInput_in = Scalar::all(0);
	myHist* pHist = new myHist(myHist::Type::Image);
	pHist->CalcHist(image);
	pHist->Show(canvasInput_in);
	return canvasInput_in;
}

Mat LBP(Mat in, int threshold) {
	const uchar LBP58[256] = { 0,1,2,3,4,0,6,7,8,0,0,0,12,0,14,15,16,0,0,0,0,0,0,0,24,0,0,0,28,0,30,31,32,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,48,0,0,0,0,0,0,0,56,0,0,0,60,0,62,63,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,96,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,112,0,0,0,0,0,0,0,120,0,0,0,124,0,126,127,128,129,0,131,0,0,0,135,0,0,0,0,0,0,0,143,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,191,192,193,0,195,0,0,0,199,0,0,0,0,0,0,0,207,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,223,224,225,0,227,0,0,0,231,0,0,0,0,0,0,0,239,240,241,0,243,0,0,0,247,248,249,0,251,252,253,254,255 };
	Mat out; out.create(in.rows, in.cols, CV_8UC1);
	for (int r = 1; r < in.rows - 1; r++)
	{
		for (int c = 1; c < in.cols - 1; c++)
		{

			uchar code = 0;
			int center = in.at<uchar>(r, c) + threshold;

			code |= (in.at<uchar>(r - 1, c - 1) >= center) << 7;
			code |= (in.at<uchar>(r - 1, c + 0) >= center) << 6;
			code |= (in.at<uchar>(r - 1, c + 1) >= center) << 5;
			code |= (in.at<uchar>(r + 0, c + 1) >= center) << 4;
			code |= (in.at<uchar>(r + 1, c + 1) >= center) << 3;
			code |= (in.at<uchar>(r + 1, c + 0) >= center) << 2;
			code |= (in.at<uchar>(r + 1, c - 1) >= center) << 1;
			code |= (in.at<uchar>(r + 0, c - 1) >= center) << 0;
			//--------------------------------------------------
			//code = LBP58[code];
			//--------------------------------------------------
			out.at<uchar>(r - 1, c - 1) = code;
		}
	}
	return out;
}

double * Hist_double(Mat image) {
	int Histogram[256] = { 0 };
	int Histogram_MAX = 1;
	for (int r = 0; r < image.rows; r++)
	{
		for (int c = 0; c < image.cols; c++)
		{
			Histogram[image.at<uchar>(r, c)]++;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		if (Histogram_MAX<Histogram[i]) {
			Histogram_MAX = Histogram[i];
		}
	}

	static double r[256];
	for (int i = 0; i < 256; i++) {
		r[i] = Histogram[i] * 1.0 / Histogram_MAX;
	}
	return r;
}

int main() {
	const string read_dir = "../input/";
	const string fileName[] = { "000.bmp" };
	for each (string fileNametemp in fileName)
	{
		Mat image = imread(read_dir + fileNametemp, 1);
		Mat image_LBP;
		imshow("原始圖像", image);
		//------------------------------------------------------------------------------------------------------------------------------
		//medianBlur(image, image, 3);
		//imshow("中值圖像", image);
		GaussianBlur(image, image, Size(3, 3), 0, 0);
		imshow("高斯圖像", image);
		//------------------------------------------------------------------------------------------------------------------------------
		vector<Mat> imageBGR_vector;
		Mat imageBGR;
		//RGB三通道分離
		split(image, imageBGR_vector);
		//求原始圖像的RGB分量的均值
		double R, G, B;
		B = mean(imageBGR_vector[0])[0];
		G = mean(imageBGR_vector[1])[0];
		R = mean(imageBGR_vector[2])[0];
		//需要調整的RGB分量的增益
		double KR, KG, KB;
		KB = (R + G + B) / (3 * B);
		KG = (R + G + B) / (3 * G);
		KR = (R + G + B) / (3 * R);
		//調整RGB三個通道各自的值
		imageBGR_vector[0] = imageBGR_vector[0] * KB;
		imageBGR_vector[1] = imageBGR_vector[1] * KG;
		imageBGR_vector[2] = imageBGR_vector[2] * KR;
		//RGB三通道圖像合併
		imageBGR_vector[0] = imageBGR_vector[0];
		imageBGR_vector[1] = imageBGR_vector[1];
		imageBGR_vector[2] = imageBGR_vector[2];
		merge(imageBGR_vector, imageBGR);
		imshow("白平衡", imageBGR);
		imwrite("../_白平衡.bmp", imageBGR);
		//------------------------------------------------------------------------------------------------------------------------------
		Mat GX; Sobel(imageBGR, GX, 0, 1, 0, 3, 1, 0, BORDER_DEFAULT);
		Mat GY; Sobel(imageBGR, GY, 0, 0, 1, 3, 1, 0, BORDER_DEFAULT);
		imageBGR = imageBGR - GX * 0.25 - GY * 0.25;
		imshow("銳化", imageBGR);
		imwrite("../_銳化.bmp", imageBGR);
		//------------------------------------------------------------------------------------------------------------------------------
		cvtColor(imageBGR, image, CV_BGR2GRAY);
		imshow("灰階", image);
		imwrite("../_灰階.bmp", image);
		//------------------------------------------------------------------------------------------------------------------------------
		image_LBP = LBP(image, 16);
		imshow("LBP", image_LBP);
		imwrite("../_LBP.bmp", image_LBP);
		//------------------------------------------------------------------------------------------------------------------------------
		Mat Hist_image_LBP = Hist(image_LBP);
		imshow("LBP_Hist", Hist_image_LBP);
		imwrite("../_LBP_Hist.bmp", Hist_image_LBP);

		system("CLS");
		double  *p = Hist_double(image_LBP);
		for (int i = 0; i < 58; i++) {
			if (i % 2 == 0)cout << "\n";
			printf("Hist_double[%3d]) : %12.8f  ", LBP_58_table[i], double(*(p + LBP_58_table[i] * sizeof(double))));
		}
		cout << "\n\n Hist_double END";
		//------------------------------------------------------------------------------------------------------------------------------
		//imageBGR//創建一個對象		
		/* Load the SLIC_image and convert to Lab colour space. */
		IplImage *SLIC_image = cvCloneImage(&(IplImage)imageBGR);
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
		Mat SLIC_image_MAT = cvarrToMat(SLIC_image);
		imwrite("../_SLIC_image.bmp", SLIC_image_MAT);
		//------------------------------------------------------------------------------------------------------------------------------
		waitKey(0);
	}
	return 0;
}
