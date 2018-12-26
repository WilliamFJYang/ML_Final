int main() {
	const string read_dir = "../input/";
	const string fileName[] = { "000.bmp" };
  Mat image = imread(read_dir + fileNametemp, 1);
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
  waitKey(0);
  return 0;
}
