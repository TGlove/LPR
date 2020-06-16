#include <iostream>
#include "highgui/highgui.hpp"
#include "core/core.hpp"
//#include "opencv2/imgproc.hpp"
#include "imgproc/imgproc.hpp"
//#include "sprise_ocr_api.h"
#include <tesseract/baseapi.h>


using namespace std;
using namespace cv;
using namespace tesseract;

Mat convertgrey(Mat rgb)
{
	Mat grey = Mat::zeros(rgb.size(), CV_8UC1);
	for (int i = 0; i < rgb.rows; i++)
	{
		for (int j = 0; j < rgb.cols * 3; j = j + 3)
		{
			grey.at<uchar>(i, j / 3) = (rgb.at<uchar>(i, j) + rgb.at<uchar>(i, j + 1) + rgb.at<uchar>(i, j + 2)) / 3;
		}
	}
	return grey;
}

Mat equlaizehistogram(Mat grey)
{
	int count[256] = { 0 };
	float prob[256] = { 0 };

	Mat newimage = Mat::zeros(grey.size(), CV_8UC1);
	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			count[grey.at<uchar>(i, j)] ++;//get the pixel number
		}
	}
	for (int k = 0; k < 256; k++)
	{
		prob[k] = float(float(count[k]) / float((grey.rows * grey.cols)));//calculate probability

	}
	for (int l = 1; l < 256; l++)
	{
		prob[l] = prob[l] + prob[l - 1];//calculate accumulate value

	}
	for (int m = 0; m < 256; m++)
	{
		prob[m] = int(255 * prob[m]);//(c-1)*accumulate value

	}
	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			newimage.at<uchar>(i, j) = prob[grey.at<uchar>(i, j)];
		}
	}
	return newimage;
}
Mat blurbetter(Mat grey)
{
	Mat newimage = Mat::zeros(grey.size(), CV_8UC1);
	for (int i = 1; i < grey.rows - 1; i++)
	{
		for (int j = 1; j < grey.cols - 1; j++)
		{
			int sum = 0;
			for (int ii = -1; ii <= 1; ii++)
			{
				for (int jj = -1; jj <= 1; jj++)
				{
					sum = sum + grey.at<uchar>(i + ii, j + jj);
				}
			}
			newimage.at<uchar>(i, j) = sum / 9;
		}
	}
	return newimage;
 }
Mat findedge(Mat blur,int a)
 {
	Mat newimage = Mat::zeros(blur.size(), CV_8UC1);
	for (int i = 1; i < blur.rows - 1; i++)
	{
		for (int j = 1; j < blur.cols - 1; j++)
		{
			int sum1 = 0;
			int sum2 = 0;
			sum1 = (blur.at<uchar>(i - 1, j - 1) + blur.at<uchar>(i, j - 1) + blur.at<uchar>(i + 1, j - 1)) / 3;
			sum2 = (blur.at<uchar>(i - 1, j + 1) + blur.at<uchar>(i, j + 1) + blur.at<uchar>(i + 1, j + 1)) / 3;

			if (sum2 - sum1 > a)
			{
				newimage.at<uchar>(i, j) = 255;
			}
		}
	}

	return newimage;
 }

Mat dilation(Mat edge, int windowsize)
{
	Mat newimage = Mat::zeros(edge.size(), CV_8UC1);
	for (int i = windowsize; i < edge.rows - windowsize; i++)
	{
		for (int j = windowsize; j < edge.cols - windowsize; j++)
		{
			for (int ii = i - windowsize; ii < i + windowsize; ii++)
			{
				for (int jj = j - windowsize; jj < j + windowsize; jj++)
				{
					if (edge.at<uchar>(ii, jj) == 255)
					{
						newimage.at<uchar>(i, j) = 255;
					}
				}
			}
		}
	}
	return newimage;
}

Mat erosion(Mat dilation, int windowsize)
{
	Mat newimage = Mat::zeros(dilation.size(), CV_8UC1);
	for (int i = windowsize; i < dilation.rows - windowsize; i++)
	{
		for (int j = windowsize; j < dilation.cols - windowsize; j++)
		{
			for (int ii = i - windowsize; ii < i + windowsize; ii++)
			{
				for (int jj = j - windowsize; jj < j + windowsize; jj++)
				{
					if (dilation.at<uchar>(ii, jj) == 0)
					{
						newimage.at<uchar>(i, j) = 0;
					}
					else
					{
						newimage.at<uchar>(i, j) = 255;
					}
				}
			}
		}
	}
	return newimage;
}

bool verifysize(Rect re)
{
	float error = 0.3;
	float aspect = 3.142857;
	int min = 1 * aspect * 1;
	int max = 2000 * aspect * 2000;

	float rmin = aspect - (aspect * error);
	float rmax = aspect + (aspect * error);

	int area = re.height * re.width;
	float r = (float)re.width / (float)re.height;
	if (r < 1)
	{
		r = (float)re.height / (float)re.width;
	}
	
	if ((area < min || area > max) || (r < rmin || r > rmax))
	{
		return false;
	}
	else
	{
		return true;
	}
	
}


int otsu(Mat plate) {
	float count[256] = { 0 };
	float prob[256] = { 0 };
	float accumul[256] = { 0 };
	for (int i = 0; i < plate.rows; i++) {
		for (int j = 0; j < plate.cols; j++) {
			count[plate.at<uchar>(i, j)]++;
		}
	}
	//probability
	for (int i = 0; i < 256; i++) {
		prob[i] = count[i] / (plate.rows * plate.cols);
	}
	//accumulative probability
	accumul[0] = prob[0];
	for (int i = 1; i < 256; i++) {
		accumul[i] = accumul[i - 1] + prob[i];
	}
	//mew
	float mue[256] = { 0 };
	for (int i = 1; i < 256; i++) {
		mue[i] = mue[i - 1] + i * prob[i];
	}
	//sigma
	float sigma[256] = { 0 };
	for (int i = 0; i < 256; i++) {
		sigma[i] = pow(mue[255] * accumul[i] - mue[i], 2) / (accumul[i] * (1 - accumul[i]));
	}
	//find max sigma
	float max = sigma[0];
	int result = 0;
	for (int i = 1; i < 256; i++) {
		if (sigma[i] > max) {
			max = sigma[i];
			result = i;
		}
	}
	return result + 30;
}

Mat convertbinary(Mat grey, int th)
{
	Mat newimage = Mat::zeros(grey.size(), CV_8UC1);
	for (int i = 0; i < grey.rows; i++)
	{
		for (int j = 0; j < grey.cols; j++)
		{
			if (grey.at<uchar>(i, j) >= th)
			{
				newimage.at<uchar>(i, j) = 255;
			}
		}
	}
	return newimage;
}


int main(){
	Mat img;
	string image[20] = { "1","2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12","13","14","15","16","17","18","19","20" };
	Rect rect_first;
	Mat plate;
	Scalar black = CV_RGB(0, 0, 0);
	RotatedRect rorect;
	for (int i = 0; i < 20; i++)
	{
		img = imread(image[i] + ".jpg");
		Mat grey = convertgrey(img);
		Mat equalhis = equlaizehistogram(grey);
		Mat b2 = blurbetter(equalhis);
		Mat finde = findedge(b2, 45);
		Mat dila = dilation(finde, 6);


		int numberofcontour = 0;
		bool flag = true;
		/*do
		{*/
		Mat blob = dila.clone();
		vector<vector<Point>> contours1;
		vector<Vec4i> hierachy1;
		findContours(dila, contours1, hierachy1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

		Mat dst = Mat::zeros(grey.size(), CV_8UC3);
		if (!contours1.empty())
		{
			for (int i = 0; i < contours1.size(); i++)
			{
				Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
				drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
				numberofcontour++;
			}
		}

		cout << "total number of contour :" << numberofcontour << endl;


		for (int i = 0; i < contours1.size(); i++)
		{
			int area = rect_first.height * rect_first.width;
			float r = float(rect_first.width) / float(rect_first.height);
			int minarea = int(area) - int(area / 100 * 6);
			int maxarea = int(area) + int(area / 100 * 6);
			rect_first = boundingRect(contours1[i]);
			double conarea = contourArea(contours1[i], false);
			if ((rect_first.width / rect_first.height) < 2.0 || rect_first.height > 45 || rect_first.width < 55 || rect_first.width < 75)
			{
				drawContours(blob, contours1, i, black, -1, 8, hierachy1);
				numberofcontour--;
			}
			else
			{
				plate = grey(rect_first);
				break;
			}
		}


		if (numberofcontour == 0)
		{

			dila = dilation(finde, 8);
			Mat blob = dila.clone();
			vector<vector<Point>> contours1;
			vector<Vec4i> hierachy1;
			findContours(dila, contours1, hierachy1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

			Mat dst = Mat::zeros(grey.size(), CV_8UC3);
			for (int i = 0; i < contours1.size(); i++)
			{
				int area = rect_first.height * rect_first.width;
				float r = float(rect_first.width) / float(rect_first.height);
				int minarea = int(area) - int(area / 100 * 6);
				int maxarea = int(area) + int(area / 100 * 6);
				rect_first = boundingRect(contours1[i]);
				double conarea = contourArea(contours1[i], false);
				if ((rect_first.width / rect_first.height) < 2.0 || rect_first.height > 45 || rect_first.width < 55 || rect_first.width < 75)
				{
					drawContours(blob, contours1, i, black, -1, 8, hierachy1);
					numberofcontour--;
				}
				else
				{
					plate = grey(rect_first);
				}
			}
		}
		else if (numberofcontour == 2)
		{
			dila = dilation(finde, 8);
			Mat blob = dila.clone();
			vector<vector<Point>> contours1;
			vector<Vec4i> hierachy1;
			findContours(dila, contours1, hierachy1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE, Point(0, 0));

			Mat dst = Mat::zeros(grey.size(), CV_8UC3);
			for (int i = 0; i < contours1.size(); i++)
			{
				int area = rect_first.height * rect_first.width;
				float r = float(rect_first.width) / float(rect_first.height);
				int minarea = int(area) - int(area / 100 * 6);
				int maxarea = int(area) + int(area / 100 * 6);
				rect_first = boundingRect(contours1[i]);
				double conarea = contourArea(contours1[i], false);
				if ((rect_first.width / rect_first.height) < 2.0 || rect_first.height > 45 || rect_first.width < 55 || rect_first.width < 85)
				{
					drawContours(blob, contours1, i, black, -1, 8, hierachy1);
					numberofcontour--;
				}
				else
				{
					plate = grey(rect_first);
				}
			}
		}

		/*	} while (flag == false);*/

		if (plate.rows != 0 && plate.cols != 0)
		{
			//imshow("plate", plate);
			int threshold = otsu(plate);
			Mat binplate = convertbinary(plate, threshold);
			imshow("binary plate", binplate);

			// pass it to tesseract api
			tesseract::TessBaseAPI tess;
			tess.Init(NULL, "eng", tesseract::OEM_DEFAULT);
			tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
			tess.SetImage((uchar*)binplate.data, binplate.cols, binplate.rows, 1, binplate.cols);

			// get the text
			char* out = tess.GetUTF8Text();
			cout << "--------------------------------------" << endl;
			std::cout << out << std::endl;
		}
		waitKey();
	}
}