// autonomous_car.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>


using namespace std;
using namespace cv;

void findRoad(Mat &img);
void myClustered(Mat &img);
void findCenter();

double width_scale = 0.8;
double height_scale = 0.2;

Point2d x1, x2;
const int K = 4;  //Kmean 1~8
//int colors[4] = { 127, 85, 42, 21 };
Mat centers; //Kmeans center
double max_u = INT_MIN, min_u = INT_MAX;
double max_v = INT_MIN, min_v = INT_MAX;

const String videoFilename = "C:\\GE\\School_project\\test_MOV\\t4.mp4";
VideoCapture cap(videoFilename);

int _tmain(int argc, _TCHAR* argv[])
{
	/*
	VideoCapture cap(0); // open the video camera no. 0

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "ERROR: Cannot open the video file" << endl;
		return -1;
	}
	*/

	if (!cap.isOpened()){
		//error in opening the video input
		cerr << "Unable to open video file: " << videoFilename << endl;
		exit(EXIT_FAILURE);
	}


	//namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);
	Mat frame;
	Mat roi;
	bool bSuccess = cap.read(frame); // read a new frame from videoMat frame;
	if (!bSuccess) return 0;


	resize(frame, frame, cv::Size(frame.cols / 2, frame.rows / 2));

	x1.y = (int)(frame.rows * (1.0 - height_scale) / 2.0);
	x1.x = (int)(frame.cols * (1.0 - width_scale) / 2.0);

	x2.y = x1.y + frame.rows * height_scale;
	x2.x = x1.x + frame.cols * width_scale;


	cout << frame.rows << " : " << frame.cols << endl;
	cout << x1.x << " : " << x1.y << endl;
	cout << x2.x << " : " << x2.y << endl;
	findCenter();
	while (1){
		bool bSuccess = cap.read(frame); // read a new frame from videoMat frame;

		if (bSuccess){

			resize(frame, frame, cv::Size(frame.cols / 2, frame.rows / 2));
			roi = frame(Rect(x1, x2)).clone();
			findRoad(roi);

			rectangle(frame, x1, x2, Scalar(200, 200, 200), 8, 8, 0);
			imshow("MyVideo", frame);
			imshow("ROI", roi);


			char c;
			c = waitKey(1);
			if (c == 27) break;
		}
		else exit(0);

	}

	return 0;
}

void findRoad(Mat &img)
{

	GaussianBlur(img, img, Size(9, 9), 0, 0);
	cvtColor(img, img, CV_BGR2YUV);
	
	Mat hsv_planes[3];
	Mat tmp = img.clone();
	split(tmp, hsv_planes);
	int histSize = 256; //from 0 to 360
	/// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	Mat h_hist, s_hist, v_hist;
	// Compute the histograms:
	calcHist(&hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&hsv_planes[1], 1, 0, Mat(), s_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&hsv_planes[2], 1, 0, Mat(), v_hist, 1, &histSize, &histRange, uniform, accumulate);
	// Draw the histograms for R, G and B
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	// Normalize the result to [ 0, histImage.rows ]
	normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(s_hist, s_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	normalize(v_hist, v_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++){
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(h_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(h_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(s_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(s_hist.at<float>(i))),
			Scalar(0, 255, 0), 2, 8, 0);
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(v_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(v_hist.at<float>(i))),
			Scalar(0, 0, 255), 2, 8, 0);
	}

	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);
	
	myClustered(img);

	//cvtColor(img, img, CV_RGB2GRAY);
	
	
	//threshold(img, img, 160, 255, THRESH_BINARY);
}

void findCenter()
{
	Mat img;
	for (int c = 0; c < 100; c++){
		bool bSuccess = cap.read(img); // read a new frame from videoMat frame;

		if (bSuccess){
			resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
			img = img(Rect(x1, x2)).clone();

			GaussianBlur(img, img, Size(9, 9), 0, 0);
			cvtColor(img, img, CV_BGR2YUV);

			Mat src;
			resize(img, src, cv::Size(img.cols / 3, img.rows / 3));
			Mat p = Mat::zeros(src.cols*src.rows, 2, CV_32F);
			Mat bestLabels, clustered;

			//Mat bgr = src.reshape(1, src.rows*src.cols);
			Mat yuv[3];
			split(src, yuv);

			for (int i = 0; i<src.cols*src.rows; i++) {
				//p.at<float>(i, 0) = yuv[0].data[i] / 255.0;
				p.at<float>(i, 0) = yuv[1].data[i] / 255.0;
				p.at<float>(i, 1) = yuv[2].data[i] / 255.0;
			}

			//bgr.convertTo(p, CV_32FC3, 1.0 / 255.0);

			kmeans(p, K, bestLabels,
				TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
				3, KMEANS_PP_CENTERS, centers);


			double tmp_u, tmp_v;
			
			for (int i = 0; i < K; i++){
				tmp_u = centers.at<float>(i, 0);
				tmp_v = centers.at<float>(i, 1);
				if (min_u > tmp_u) min_u = tmp_u;
				if (max_u < tmp_u) max_u = tmp_u;
				if (min_v > tmp_v) min_v = tmp_v;
				if (max_v < tmp_v) max_v = tmp_v;
				cout << tmp_u << " : " << tmp_v << endl;
			}
			cout << "MAX_U: " << max_u << " -- MIN_U: " << min_u << endl;
			cout << "MAX_V: " << max_v << " -- MIN_V: " << min_v << endl;
			cout << "----------------------------------" << endl;

		}
		else exit(0);
		
	}

}

void myClustered(Mat &img)
{
	Mat src;
	resize(img, src, cv::Size(img.cols / 3, img.rows / 3));
	Mat p = Mat::zeros(src.cols*src.rows, 2, CV_32F);
	Mat bestLabels, clustered;

	//Mat bgr = src.reshape(1, src.rows*src.cols);
	Mat yuv[3];
	split(src, yuv);

	for (int i = 0; i<src.cols*src.rows; i++) {
		//p.at<float>(i, 0) = yuv[0].data[i] / 255.0;
		p.at<float>(i, 0) = yuv[1].data[i] / 255.0;
		p.at<float>(i, 1) = yuv[2].data[i] / 255.0;
	}

	//bgr.convertTo(p, CV_32FC3, 1.0 / 255.0);

	cout << "p is of size: " << p.rows << "x" << p.cols << endl;

	//KMEANS_USE_INITIAL_LABELS   KMEANS_PP_CENTERS   KMEANS_RANDOM_CENTERS

	kmeans(p, K, bestLabels,
		TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
		3, KMEANS_PP_CENTERS, centers);
	cout << "MAX_U: " << max_u << " -- MIN_U: " << min_u << endl;
	cout << "MAX_V: " << max_v << " -- MIN_V: " << min_v << endl;
	cout << centers << endl;
	cout << "----------------------------------" << endl;

	clustered = Mat(src.rows, src.cols, CV_32F);
	int cntR = 0, cntNR = 0;
	double tmp_u, tmp_v;
	int arr[4] = { 127, 85, 42, 21 };
	int colors[K] = {0};

	for (int i = 0; i<K; i++) {
		tmp_u = centers.at<float>(i, 0);
		tmp_v = centers.at<float>(i, 1);
		if (tmp_u <= max_u + 0.01 && tmp_u >= min_u - 0.01){
			if (tmp_v <= max_v + 0.01  && tmp_v >= min_v - 0.01) colors[i] = 255;
			else colors[i] = 0;
		}
		else colors[i] = 0;
	}

	for (int i = 0; i<src.cols*src.rows; i++) {
		clustered.at<float>(i / src.cols, i%src.cols) = (float)(colors[bestLabels.at<int>(0, i)]);
	}
	/*
	double t = (double)cntR / (cntR + cntNR);
	double tt = (double)cntNR / (cntR + cntNR);
	if (t >= 0.7) threshold(clustered, clustered, 128, 255, THRESH_BINARY);
	else threshold(clustered, clustered, 128, 255, THRESH_BINARY_INV);
	*/
	clustered.convertTo(clustered, CV_8U);

	/*
	for (int i = 0; i<3; i++) {
		circle(clustered, Point(centers.at<float>(i, 0)*255, centers.at<float>(i, 1)*255), 4, Scalar(0));
		cout << Point(centers.at<float>(i, 0)*255, centers.at<float>(i, 1)*255) << endl;
	}
	*/
	cout << clustered.rows << " : " << clustered.cols << endl;
	
	//resize(centers, centers, cv::Size(centers.cols * 30, centers.rows * 30));
	//imshow("center", centers);
	imshow("clustered", clustered);
}


