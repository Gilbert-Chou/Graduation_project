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

double width_scale = 0.8;
double height_scale = 0.2;


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
	String videoFilename = "C:\\GE\\School_project\\test_MOV\\t1.mp4";
	VideoCapture cap(videoFilename);
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
	Point2d x1, x2;

	resize(frame, frame, cv::Size(frame.cols / 2, frame.rows / 2));

	x1.y = (int)(frame.rows * (1.0 - height_scale) / 2.0);
	x1.x = (int)(frame.cols * (1.0 - width_scale) / 2.0);

	x2.y = x1.y + frame.rows * height_scale;
	x2.x = x1.x + frame.cols * width_scale;


	cout << frame.rows << " : " << frame.cols << endl;
	cout << x1.x << " : " << x1.y << endl;
	cout << x2.x << " : " << x2.y << endl;


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

void findRoad(Mat &img){
	GaussianBlur(img, img, Size(9, 9), 0, 0);
	cvtColor(img, img, CV_BGR2YUV);
	//GaussianBlur(img, img, Size(9, 9), 0, 0);
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



	//cvtColor(img, img, CV_RGB2GRAY);
	
	
	//threshold(img, img, 160, 255, THRESH_BINARY);
}

