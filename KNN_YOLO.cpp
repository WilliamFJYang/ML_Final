#include "stdafx.h"
#include "myHist.h"
#include "slic.h"

#include <cstdio>
#include <direct.h>
#include <float.h>
#include <fstream>
#include <io.h>
#include <iostream>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc/slic.hpp>
#include <sstream>
#include <stdio.h>
#include <string.h>
#include <tchar.h>
#include <vector>
#include <Windows.h>

using namespace std;
using namespace cv;
using namespace cv::ml;
using namespace cv::dnn;
using namespace cv::ximgproc;
//----

const String training_dir_list[] = {
	"..\\input\\FAAA/*.jpg",
	"..\\input\\1/*.jpg" ,
	"..\\input\\2/*.jpg" ,
	"..\\input\\3/*.jpg" ,
	"..\\input\\4/*.jpg"
};
const int training_dir_lable_list[] = {
	0,
	1,
	2,
	3,
	4
};
const String Lable_list[] = {
	"" ,
	"R",//"Reindeer" ,
	"M",//"Magnet" ,
	"C",//"Chopper " ,
	"E" //"Elephant"
};
const String predict_dir_list[] = {
	"..\\input/*.jpg"
};
const char TXT_filename_LBP58[] = "..\\LBP58_out.txt";
const char TXT_filename_Label[] = "..\\Label_out.txt";
const char TXT_filename_predict[] = "..\\predict_out.txt";

const float confThreshold = 0.5;
const float nmsThreshold = 0.4;

const string KNN_Path = "../knn.xml";
const int KNN_K = 9;

//const int LBP_58_table[58] = { 0,128,192,224,240,248,252,254,64,96,112,120,124,126,254,32,48,56,60,62,190,254,16,24,28,30,158,222,254,8,12,14,142,206,238,254,4,6,134,198,230,246,254,2,130,194,226,242,250,254,128,192,224,240,248,252,254,255 };
const int LBP_11_table[11] = { 0,16,24,28,30,8,12,14,4,6,2 };

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
			out.at<uchar>(r - 1, c - 1) = code;
		}
	}
	return out;
}

double * Hist_double(Mat image) {
	double Histogram[256] = { 0 };
	double Histogram_MAX = 1;
	for (int r = 0; r < image.rows; r++)
	{
		for (int c = 0; c < image.cols; c++)
		{
			Histogram[image.at<uchar>(r, c)]++;
		}
	}

	for (int i = 0; i < 256; i++)
	{
		Histogram[i] = pow(Histogram[i], 1.0 / 2);
	}

	for (int i = 0; i < 256; i++)
	{
		if (Histogram_MAX < Histogram[i]) {
			Histogram_MAX = Histogram[i];
		}
	}

	static double r[256];
	for (int i = 0; i < 256; i++) {
		r[i] = Histogram[i] * 1.0 / Histogram_MAX;
	}
	return r;
}

Mat White_balance(Mat image) {
	vector<Mat> image_vector;
	Mat out;
	//RGB三通道分離
	split(image, image_vector);
	//求原始圖像的RGB分量的均值
	double R, G, B;
	B = mean(image_vector[0])[0];
	G = mean(image_vector[1])[0];
	R = mean(image_vector[2])[0];
	//需要調整的RGB分量的增益
	double KR, KG, KB;
	KB = (R + G + B) / (3 * B);
	KG = (R + G + B) / (3 * G);
	KR = (R + G + B) / (3 * R);
	//調整RGB三個通道各自的值
	image_vector[0] = image_vector[0] * KB;
	image_vector[1] = image_vector[1] * KG;
	image_vector[2] = image_vector[2] * KR;
	//RGB三通道圖像合併
	image_vector[0] = image_vector[0];
	image_vector[1] = image_vector[1];
	image_vector[2] = image_vector[2];
	merge(image_vector, out);
	return out;
}

void split(const string& s, vector<string>& sv, const char delim = ' ') {
	sv.clear();
	istringstream iss(s);
	string temp;

	while (getline(iss, temp, delim)) {
		sv.emplace_back(move(temp));
	}
	return;
	//https://liam.page/2017/08/03/split-a-string-in-Cpp/
}

double IOU(const Rect& r1, const Rect& r2)
{
	int x1 = max(r1.x, r2.x);
	int y1 = max(r1.y, r2.y);
	int x2 = min(r1.x + r1.width, r2.x + r2.width);
	int y2 = min(r1.y + r1.height, r2.y + r2.height);
	int w = max(0, (x2 - x1 + 1));
	int h = max(0, (y2 - y1 + 1));
	double inter = w * h;
	double o = inter / (r1.area() + r2.area() - inter);
	return (o >= 0) ? o : 0;
	// -------------------- -
	// 	作者：DuinoDu
	// 	來源：CSDN
	// 	原文：https ://blog.csdn.net/DuinoDu/article/details/61651390 
	// 版權聲明：本文為博主原創文章，轉載請附上博文連結！
}
//NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
void nms(vector<Rect>& boxes, vector<int>& lable, vector<float>& confidences, const float confThreshold, const float nmsThreshold, std::vector<int>& indices)
{
	vector<int> new_proposals;
	for (int i = 0; i < boxes.size(); i++) {
		boolean flag = true;
		for (int j = 0; j < boxes.size(); j++) {
			if (lable[i] == lable[j]) {
				if (IOU(boxes[i], boxes[i]) > confThreshold) {
					//if (confidences[i] < confidences[j]) {
					//	flag = false;
					//}
					if (boxes[i].area() < boxes[j].area()) {
						flag = false;
					}
				}
			}
		}
		if (flag) {
			new_proposals.push_back(i);
		}
	}
	indices = new_proposals;
}

int main() {

	int training_data_number = 0;
	int predict_data_number = 0;
	int training_data_cnt = 0;
	int training_dir_cnt = 0;
	Mat training_Mat_Data;
	Mat training_Mat_lable;
	Mat testMat; testMat.create(1, 22, CV_32FC1);

	//--------------------------------------------------------KNN
	int cnt = 0;
	//--------------------------------------------------------main
	printf("filename sample : 0_X^120Y^240.jpg\n");
	printf("相對目標左上角座標  X^___Y^___    \n");
	cout << "\n training_dir : ";
	_mkdir("..\\input");
	_mkdir("..\\temp");
	for each (string read_dir in training_dir_list)
	{
		printf("\n %3d : ", training_dir_lable_list[cnt++]);
		try { read_dir = read_dir.replace(read_dir.find("/*.jpg"), 6, ""); }
		catch (const exception&) {}
		try { read_dir = read_dir.replace(read_dir.find("/*.bmp"), 6, ""); }
		catch (const exception&) {}
		_mkdir(read_dir.c_str());
		cout << "\n      input:" << read_dir;

		try { read_dir = read_dir.replace(read_dir.find("input"), 5, "temp"); }
		catch (const exception&) {}
		_mkdir(read_dir.c_str());
		cout << "\n      temp :" << read_dir;

	}
	for each (String read_dir in training_dir_list)//圖片總量
	{
		try
		{
			vector<String> fileName_list;
			glob(read_dir, fileName_list);
			for each (String fileName in fileName_list) {
				training_data_number++;
			}
		}
		catch (const exception&)
		{
		}
	}
	cout << "\n----------------------------------------training data : " << training_data_number;
	cout << "\n predict_dir : ";
	_mkdir("..\\predict");
	for each (string read_dir in predict_dir_list)
	{
		try { read_dir = read_dir.replace(read_dir.find("/*.jpg"), 6, ""); }
		catch (const exception&) {}
		try { read_dir = read_dir.replace(read_dir.find("/*.bmp"), 6, ""); }
		catch (const exception&) {}
		_mkdir(read_dir.c_str());

		cout << "\n      input  :" << read_dir;
		try { read_dir = read_dir.replace(read_dir.find("input"), 5, "predict"); }
		catch (const exception&) {}
		_mkdir(read_dir.c_str());
		cout << "\n      predict:" << read_dir;
	}
	for each (String read_dir in predict_dir_list)//圖片總量
	{
		try
		{
			vector<String> fileName_list;
			glob(read_dir, fileName_list);
			for each (String fileName in fileName_list) {
				predict_data_number++;
			}
		}
		catch (const exception&)
		{
		}
	}
	cout << "\n----------------------------------------predict  data : " << predict_data_number;
	bool training_yes_no = true;
	if (FILE * file = fopen(KNN_Path.c_str(), "r"))//有KNN檔案
	{
		fclose(file);
		training_yes_no = (MessageBox(NULL, L"KNN sample?\n", L"Yes or No", MB_ICONEXCLAMATION | MB_YESNO) == IDYES);
	}

	bool nms_in_yes_no = (MessageBox(NULL, L"NMS data in show?\n", L"Yes or No", MB_ICONEXCLAMATION | MB_YESNO) == IDYES);

	if (training_yes_no) {

		training_Mat_lable.create(training_data_number, 1, CV_32SC1);//訓練特徵
		training_Mat_Data.create(training_data_number, 22, CV_32FC1);//訓練特徵

		fstream LBP58_out;
		fstream Label_out;
		LBP58_out.open(TXT_filename_LBP58, ios::out);
		Label_out.open(TXT_filename_Label, ios::out);
		for each (String read_dir in training_dir_list)
		{
			try
			{
				vector<String> fileName_list;
				glob(read_dir, fileName_list);
				for each (String fileName in fileName_list) {
					cout << "\n read : " << fileName;
					string fileName_temp = fileName;
					try { fileName_temp = fileName_temp.replace(fileName_temp.find("input"), 5, "temp"); }
					catch (const exception&) {}
					try { fileName_temp = fileName_temp.replace(fileName_temp.find(".jpg"), 4, ""); }
					catch (const exception&) {}
					try { fileName_temp = fileName_temp.replace(fileName_temp.find(".bmp"), 4, ""); }
					catch (const exception&) {}
					String write_fileName = fileName_temp;
					cout << "\nwrite : " << write_fileName;
					int dX = 0;
					int dY = 0;
					try
					{
						dX = stoi(fileName_temp.substr(fileName_temp.find("X^") + 2, fileName_temp.find("Y^") - fileName_temp.find("X^") - 2));
						dY = stoi(fileName_temp.substr(fileName_temp.find("Y^") + 2, fileName_temp.size() - fileName_temp.find("Y^") - 2));
						cout << "\n^X : " << dX;
						cout << "\n^Y : " << dY;
					}
					catch (const exception&)
					{
					}

					Mat image = imread(fileName, 1);
					if (!image.data) {
						cout << "\n" << fileName << " : no data...\n\n";
						continue;
					}
					//imshow("原始圖像", image);
					imwrite(write_fileName + "_00原始圖像.jpg", image);
					//------------------------------------------------------------------------------------------------------------------------------
					GaussianBlur(image, image, Size(3, 3), 0, 0);
					//imshow("高斯圖像", image);
					imwrite(write_fileName + "_01高斯圖像.jpg", image);
					//------------------------------------------------------------------------------------------------------------------------------
					image = White_balance(image);
					//imshow("White_balance", image);
					imwrite(write_fileName + "_02White_balance.jpg", image);
					//------------------------------------------------------------------------------------------------------------------------------
					Mat GX;
					Mat GY;
					Sobel(image, GX, 0, 1, 0, 3, 1, 0, BORDER_DEFAULT);
					Sobel(image, GY, 0, 0, 1, 3, 1, 0, BORDER_DEFAULT);
					image = image - GX * 0.25 - GY * 0.25;
					//imshow("銳化", image);
					imwrite(write_fileName + "_03銳化.jpg", image);
					//------------------------------------------------------------------------------------------------------------------------------
					cvtColor(image, image, CV_BGR2GRAY);
					//imshow("灰階", image);
					imwrite(write_fileName + "_04灰階.jpg", image);
					//------------------------------------------------------------------------------------------------------------------------------
					Mat image_LBP;
					image_LBP = LBP(image, 4);
					//imshow("LBP", image_LBP);
					imwrite(write_fileName + "_05LBP.jpg", image_LBP);
					//------------------------------------------------------------------------------------------------------------------------------
					Mat image_LTP;
					image_LTP = LBP(256 - image, 4);
					//imshow("LTP", image_LTP);
					imwrite(write_fileName + "_05LTP.jpg", image_LTP);
					//------------------------------------------------------------------------------------------------------------------------------
					double  *p_LBP = Hist_double(image_LBP);
					double  *p_LTP = Hist_double(image_LTP);

					Label_out << fileName << "\n";//寫入字串
					for (int i = 0; i < 11; i++) {
						if (i % 2 == 0)cout << "\n";
						printf("Hist_LBP[%3d]) : %12.8f  ", LBP_11_table[i], double(*(p_LBP + LBP_11_table[i] * sizeof(double))));
						if (i != 0) {
							LBP58_out << ",";
						}
						LBP58_out << fixed << setprecision(10) << double(*(p_LBP + LBP_11_table[i] * sizeof(double)));//寫入字串
						training_Mat_Data.at<float>(training_data_cnt, i) = double(*(p_LBP + LBP_11_table[i] * sizeof(double)));
					}
					for (int i = 0; i < 11; i++) {
						if (i % 2 == 0)cout << "\n";
						printf("Hist_LTP[%3d]) : %12.8f  ", LBP_11_table[i], double(*(p_LTP + LBP_11_table[i] * sizeof(double))));
						LBP58_out << ",";
						LBP58_out << fixed << setprecision(10) << double(*(p_LTP + LBP_11_table[i] * sizeof(double)));//寫入字串
						training_Mat_Data.at<float>(training_data_cnt, 11 + i) = double(*(p_LTP + LBP_11_table[i] * sizeof(double)));
					}
					//training_Mat_lable.at<int>(training_data_cnt, 0) = training_dir_lable_list[training_dir_cnt];
					training_Mat_lable.at<int>(training_data_cnt, 0) = dX * 10000 * 10 + dY * 10 + training_dir_lable_list[training_dir_cnt];
					LBP58_out << "\n";//寫入字串					   
					cout << "\n\nHist_double END";
					//------------------------------------------------------------------------------------------------------------------------------
					training_data_cnt++;
					waitKey(1);
				}
			}
			catch (const exception&)
			{

			}
			training_dir_cnt++;
		}
		LBP58_out.close();//關閉檔案
		Label_out.close();//關閉檔案
						  //WinExec("notepad.exe ..\\LBP58_out.txt", SW_SHOW);
						  //WinExec("notepad.exe ..\\Label_out.txt", SW_SHOW);
		cout << "\n\n訓練採樣完成\n\n";
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

		Ptr<KNearest> kclassifier = KNearest::create();
		Ptr<TrainData> trainData;
		trainData = TrainData::create(training_Mat_Data, SampleTypes::ROW_SAMPLE, training_Mat_lable);
		kclassifier->setIsClassifier(true);
		kclassifier->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
		kclassifier->setDefaultK(1);
		kclassifier->train(trainData);
		kclassifier->save(KNN_Path);//會把trainDataMat的原始資料全部保存為*.xml文件  

		cout << "\n\n預測模型完成\n\n";
		//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
		int cnt = 0;
		int cnt_list[5] = { 0 };
		int cnt_NUM[5] = { 0 };

		Ptr<KNearest> testModel = StatModel::load<KNearest>(KNN_Path);
		for (int i = 0; i < training_data_cnt; i++)
		{
			for (int j = 0; j < 22; j++) {
				testMat.at<float>(0, j) = training_Mat_Data.at<float>(i, j);
			}
			Mat matResults(0, 0, CV_32FC1);//保存測試結果  
			testModel->findNearest(testMat, KNN_K, matResults);//knn分類預測  
			int response = (int)matResults.at<float>(0, 0) % 10;
			double score = 0.0;
			for (int i = 1; i <= KNN_K; i++)
			{
				testModel->findNearest(testMat, i, matResults);//knn分類預測  
				int response_temp = (int)matResults.at<float>(0, 0) % 10;
				if (response_temp == response) {
					score++;
				}
			}
			score = score / KNN_K;
			cout << "Response(" << format("%.4f", score) << "):" << training_Mat_lable.at<int>(i, 0) << "(" << response << ")\n";
			if (training_Mat_lable.at<int>(i, 0) % 10 == response) {
				cnt++;
				cnt_list[training_Mat_lable.at<int>(i, 0) % 10]++;
			}
			cnt_NUM[training_Mat_lable.at<int>(i, 0) % 10]++;
		}
		cout << "\n" << cnt << "/" << training_data_cnt << "=" << 100.00* cnt / training_data_cnt << "%\n\n";
		for (int i = 0; i < 5; i++) {
			cout << cnt_list[i] << "/" << cnt_NUM[i] << "=" << fixed << setprecision(8) << cnt_list[i] * 1.0 / cnt_NUM[i] << "\n";
		}

		cout << "\n\n內部測試完成\n\n";
		MessageBox(NULL, L"KNN sample end.", L"MessageBox", MB_ICONEXCLAMATION | MB_OK);
		//system("PAUSE");
	}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	{
		fstream predict_out;
		predict_out.open(TXT_filename_predict, ios::out);
		Ptr<KNearest> testModel = StatModel::load<KNearest>(KNN_Path);
		{

			for each (String read_dir in predict_dir_list)
			{
				try
				{
					vector<String> fileName_list;
					glob(read_dir, fileName_list);
					for each (String fileName in fileName_list) {
						cout << "\n read : " << fileName << "\n";
						Mat image = imread(fileName, 1);
						if (!image.data) {
							continue;
						}
						resize(image, image, Size(720, 720 * image.rows / image.cols));
						//------------------------------------------------------------------------------------------------------------------------------
						GaussianBlur(image, image, Size(3, 3), 0, 0);
						//------------------------------------------------------------------------------------------------------------------------------
						image = White_balance(image);
						//------------------------------------------------------------------------------------------------------------------------------
						Mat GX;
						Mat GY;
						Sobel(image, GX, 0, 1, 0, 3, 1, 0, BORDER_DEFAULT);
						Sobel(image, GY, 0, 0, 1, 3, 1, 0, BORDER_DEFAULT);
						image = image - GX * 0.25 - GY * 0.25;
						//------------------------------------------------------------------------------------------------------------------------------
						imwrite("temp.jpg", image);
						//------------------------------------------------------------------------------------------------------------------------------
						Mat image_BGR = image;
						cvtColor(image, image, CV_BGR2GRAY);
						//------------------------------------------------------------------------------------------------------------------------------
						Mat image_LBP;
						image_LBP = LBP(image, 4);
						//------------------------------------------------------------------------------------------------------------------------------
						Mat image_LTP;
						image_LTP = LBP(256 - image, 4);
						//------------------------------------------------------------------------------------------------------------------------------
						string system_call = "python Superpixel_GTXT.py -l temp.jpg";
						system(system_call.c_str());

						ifstream in("news.txt");
						ostringstream tmp;
						tmp << in.rdbuf();
						string str = tmp.str();
						vector<string> strs;
						split(str, strs, '\n');
						std::vector<Rect> boxes;
						std::vector<float> confidences;
						std::vector<int> lable;
						for each (string var in strs)
						{
							int central_x = 0;
							int central_y = 0;
							central_x = stoi(var.substr(var.find("X") + 1, var.find("Y") - var.find("X") - 1));
							central_y = stoi(var.substr(var.find("Y") + 1, var.size() - var.find("Y") - 1));
							printf("\nX=%4d,Y=%4d", central_x, central_y);


							//------------------------------------------------------------------------------------------------------------------------------
							for (int grid_size = 120; grid_size < image_LTP.cols & grid_size < image_LTP.rows; grid_size *= 1.5)
							{
								if (central_x - grid_size / 2 >= 0 & central_y - grid_size / 2 >= 0 & central_x + grid_size / 2 <= image_LTP.cols& central_y + grid_size / 2 <= image_LTP.rows) {
									double  *p_LBP = Hist_double(image_LBP(Rect(central_x - grid_size / 2, central_y - grid_size / 2, grid_size, grid_size)));
									double  *p_LTP = Hist_double(image_LTP(Rect(central_x - grid_size / 2, central_y - grid_size / 2, grid_size, grid_size)));

									for (int i = 0; i < 11; i++) {
										testMat.at<float>(0, i) = double(*(p_LBP + LBP_11_table[i] * sizeof(double)));
									}
									for (int i = 0; i < 11; i++) {
										testMat.at<float>(0, 11 + i) = double(*(p_LTP + LBP_11_table[i] * sizeof(double)));
									}
									//------------------------------------------------------------------------------------------------------------------------------
									Mat matResults(0, 0, CV_32FC1);//保存測試結果  
									testModel->findNearest(testMat, KNN_K, matResults);//knn分類預測  
									int response = (int)matResults.at<float>(0, 0) % 10;
									double score = 0.0;
									for (int i = 1; i <= KNN_K; i++)
									{
										testModel->findNearest(testMat, i, matResults);//knn分類預測  
										int response_temp = (int)matResults.at<float>(0, 0) % 10;
										if (response_temp == response) {
											score++;
										}
									}
									score = score / KNN_K;
									cout << "\n                   Response(" << format("%.4f", score) << "):";
									for (int i = 0; i <= response; i++)
									{
										cout << " ";
									}
									cout << response;
									predict_out << fileName << " : " << response << "\n";//寫入字串

									if (response != 0 && score >= nmsThreshold) {
										int dX = (int)matResults.at<float>(0, 0) / 10 / 10000 % 10000;
										int dY = (int)matResults.at<float>(0, 0) / 10 % 10000;
										int TOP_L_x = central_x - dX;
										int TOP_L_y = central_y - dY;
										int Bot_R_x = central_x + grid_size / 2;
										int Bot_R_y = central_y + grid_size / 2;
										if (TOP_L_x < 0)TOP_L_x = 0;
										if (TOP_L_y < 0)TOP_L_y = 0;
										if (Bot_R_x > image.cols)TOP_L_x = Bot_R_x = image.cols;
										if (Bot_R_y > image.rows)TOP_L_y = Bot_R_y = image.rows;

										lable.push_back(response);
										boxes.push_back(Rect(Point(TOP_L_x, TOP_L_y), Size((Bot_R_x - TOP_L_x), (Bot_R_y - TOP_L_y))));
										confidences.push_back(score);
									}
								}
							}
						}
						//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
						cout << "\n非極大值抑制\n";
						vector<int> indices;
						try
						{
							nms(boxes, lable, confidences, confThreshold, nmsThreshold, indices);

							if (nms_in_yes_no) {
								for (int i = 0; i < boxes.size(); i++)
								{
									Rect box = boxes[i];
									Scalar Scalar_table[10] = { Scalar(0, 0, 0) ,Scalar(127, 0, 0) ,Scalar(0, 127, 0) ,Scalar(0, 0, 127) , Scalar(127, 127, 0), Scalar(127, 0, 127) ,Scalar(0, 127, 127) ,Scalar(255, 0, 0) ,Scalar(0, 255, 0) ,Scalar(0, 0, 255) };
									rectangle(image_BGR, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar_table[lable[i]], 1, 8, 0);
								}
							}

							for each (int i in indices)
							{
								Rect box = boxes[i];
								Scalar Scalar_table[10] = { Scalar(0, 0, 0) ,Scalar(127, 0, 0) ,Scalar(0, 127, 0) ,Scalar(0, 0, 127) , Scalar(127, 127, 0), Scalar(127, 0, 127) ,Scalar(0, 127, 127) ,Scalar(255, 0, 0) ,Scalar(0, 255, 0) ,Scalar(0, 0, 255) };
								rectangle(image_BGR, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar_table[lable[i]], 3, 8, 0);
								rectangle(image_BGR, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(255, 255, 255), 1, 8, 0);
								putText(image_BGR, Lable_list[lable[i]] + format("(%.2f)", confidences[i]), Point(box.x, box.y + 25), 0, 1, Scalar_table[lable[i]], 3);
								putText(image_BGR, Lable_list[lable[i]] + format("(%.2f)", confidences[i]), Point(box.x, box.y + 25), 0, 1, Scalar(255, 255, 255), 1);
							}
							imshow("response", image_BGR);
							waitKey(1);

							string fileName_temp = fileName;
							try { fileName_temp = fileName_temp.replace(fileName_temp.find("input"), 5, "predict"); }
							catch (const exception&) {}
							try { fileName_temp = fileName_temp.replace(fileName_temp.find(".bmp"), 4, ".jpg"); }
							catch (const exception&) {}
							String write_fileName = fileName_temp;

							cout << "\npredict : " << write_fileName;
							imwrite(write_fileName, image_BGR);
						}
						catch (const std::exception&)
						{

						}
						//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
					}
				}
				catch (const exception&)
				{

				}
			}
		}
		predict_out.close();//關閉檔案
							//WinExec("notepad.exe ..\\predict_out.txt", SW_SHOW);
		cout << "\n\n預測判斷完成\n\n";
	}
	//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	system("PAUSE");
	return 0;
}
