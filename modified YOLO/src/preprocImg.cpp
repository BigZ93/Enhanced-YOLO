#define _USE_MATH_DEFINES
#include "preprocImg.h"
#include<stdio.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include<math.h>
#include<string.h>

void mirrorImg(char *path) {
	cv::Mat originalPhoto;
	originalPhoto = cv::imread(path);
	if (originalPhoto.empty()) {
		printf("no file found\n");
	}
	cv::Mat modifiedPhoto;
	flip(originalPhoto, modifiedPhoto, 1);
	imwrite("mirror.jpg", modifiedPhoto);
}

void rotatedImg(char *path) {
	cv::Mat originalPhoto;
	originalPhoto = cv::imread(path);
	if (originalPhoto.empty()) {
		printf("no file found\n");
	}
	float height, width;
	height = originalPhoto.rows;
	width = originalPhoto.cols;
	cv::Point2f center;
	center.x = width / 2;
	center.y = height / 2;
	float angle = 5.0;
	cv::Size size;
	size = originalPhoto.size();

	//---adding black margins to the image---
	//---non enlarged might make calculations easier---
	cv::Size sizeScaled;
	float x1, x2, y1, y2;
	x1 = height * sin(angle * M_PI / 180);
	y1 = height * cos(angle * M_PI / 180);
	x2 = width * cos(angle * M_PI / 180);
	y2 = width * sin(angle * M_PI / 180);
	sizeScaled.height = (int)(y1 + y2);
	sizeScaled.width = (int)(x1 + x2);

	cv::Mat originalScaled;
	int top, bottom, left, right;
	top = (int)((sizeScaled.height - size.height) / 2);
	bottom = top;
	left = (int)((sizeScaled.width - size.width) / 2);
	right = left;
	copyMakeBorder(originalPhoto, originalScaled, top, bottom, left, right, cv::BORDER_CONSTANT);

	//---rotating the image---
	float heightScaled, widthScaled;
	heightScaled = originalScaled.rows;
	widthScaled = originalScaled.cols;
	cv::Point2f centerScaled;
	centerScaled.x = widthScaled / 2;
	centerScaled.y = heightScaled / 2;
	cv::Mat rtMatScaled;
	rtMatScaled = getRotationMatrix2D(centerScaled, -angle, 1);

	cv::Mat modifiedPhoto;
	warpAffine(originalScaled, modifiedPhoto, rtMatScaled, sizeScaled, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
	imwrite("rotated.jpg", modifiedPhoto);
}