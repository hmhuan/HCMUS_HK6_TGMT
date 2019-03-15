#include "opencv2\core.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

bool RGB2GrayScale(const Mat & sourceImage, Mat & destinationImage);
bool ChangeBrightness(const Mat & sourceImage, Mat & destinationImage, float b);
bool ChangeContrast(const Mat & sourceImage, Mat & destinationImage, float c);

int main(int argc, char * argv[])
{
	Mat srcImage, dstImage; // ma tran srcImage luu anh nguon va ma tran dstImage luu anh ket qua
	int maLenh = atoi(argv[2]); // ma lenh thuc hien 0, 1 hay 2
	float thamSo; // tham so neu ma lenh la 1 hoac 2
	bool check; // bien kiem tra
	srcImage = imread(argv[1], -1); // Doc anh tu duong dan file anh  
	if (!srcImage.data)  
	{   
		// Kiem tra xem anh co mo duoc khong
		cout << "Khong the mo anh" << endl;  
		return -1; 
	} 
	imshow("Source Image", srcImage); // show anh nguon
	// Quy uoc ma lenh
	// 0: Doi anh mau sang anh xam (neu la anh xam thi khong thay doi)
	// 1: Thay doi do sang cua anh
	// 2: Thay doi do tuong phan cua anh
	switch (maLenh)
	{
	case 0:
		check = RGB2GrayScale(srcImage, dstImage);
		break;
	case 1:
		thamSo = atof(argv[3]);
		check = ChangeBrightness(srcImage, dstImage, thamSo);
		break;
	case 2:
		thamSo = atof(argv[3]);
		check = ChangeContrast(srcImage, dstImage, thamSo);
		break;
	default:
		check = false;
		break;
	}

	if (check)
	{
		cout << "Thuc hien bien doi anh thanh cong\n";
		imshow("Destination Image", dstImage); // Show anh dich neu chuyen thanh cong
	}
	else
		cout << "Khong thuc hien duoc\n";
	waitKey(0);
	return 0; 
}

// Ham chuyen anh mau RGB sang anh trang den
bool RGB2GrayScale(const Mat & sourceImage, Mat & destinationImage)
{
	//ảnh nguồn đã là ảnh xám
	if (sourceImage.channels() == 1)
		return 0;
	//Tạo bảng lookup
	uchar lookup[256];
	for (int i = 0; i < 256; i++)
		lookup[i] = saturate_cast<uchar>(i);
	//width là chiều rộng ảnh (số cột pixel), height là chiều cao ảnh (số dòng pixels)
	int width = sourceImage.cols, height = sourceImage.rows;
	//nChannels là số kênh màu của ảnh
	int nChannels = sourceImage.channels();
	//widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
	int widthStep = sourceImage.step[0]; //so buoc cho anh nguon

	//Tạo ảnh đích với kích thước ảnh nguồn và type là ảnh grayscale
	destinationImage.create(height, width, CV_8UC1);

	//pData là con trỏ quản lý vùng nhớ ảnh đích, psData là con trỏ quản lý vùng nhớ ảnh nguồn
	uchar* pData = (uchar*)destinationImage.data;
	uchar* psData = (uchar*)sourceImage.data;

	float Y;
	for (int i = 0; i < height; i++, psData += widthStep, pData += destinationImage.step[0]) {
		uchar * pRow = pData;
		uchar * psRow = psData;
		for (int j = 0; j < width; j++, psRow += nChannels, pRow += 1) {
			//Y ← 0.299⋅R+0.587⋅G+0.114⋅B
			//Y = 0.299 * (float)psRow[2] + 0.587 * (float)psRow[1]; + 0.114 * (float)psRow[0];
			Y = ((int)psRow[2] * 2 + (int)psRow[1] * 5 + (int)psRow[0] * 1) / 8;
			pRow[0] = lookup[(int)Y];
		}
	}
	//Kiểm tra việc tạo ảnh có thành công hay không
	if (destinationImage.empty())
		return false;
	return true;
}

// Ham thay doi do sang cua anh
// Input: anh nguon, anh dich, tham so
// Output: thuc hien duoc hay khong? (true/ false)
bool ChangeBrightness(const Mat & sourceImage, Mat & destinationImage, float b)
{
	// Kiem tra anh nguon
	if (sourceImage.empty())
		return 0;
	//Tạo bảng lookup
	uchar lookup[256];
	for (int i = 0; i < 256; i++)
		lookup[i] = saturate_cast<uchar>(i + b);

	//Khởi tạo ảnh đích có kích thước và type giống ảnh nguồn
	destinationImage.create(sourceImage.rows, sourceImage.cols, sourceImage.type());

	//width là chiều rộng ảnh, height là chiều cao ảnh.
	int width = sourceImage.cols, height = sourceImage.rows;
	//nChannels là số kênh màu
	int nChannels = sourceImage.channels();
	//widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
	int widthStep = sourceImage.step[0];
	//pData là con trỏ quản lý vùng nhớ ảnh
	uchar* pData = (uchar*)destinationImage.data; //COn tro data cua anh dich
	uchar* psData = (uchar*)sourceImage.data; //COn tro data cua anh nguon
	for (int i = 0; i < height; i++, psData += widthStep, pData += widthStep) {
		uchar * pRow = pData; //Con tro dong cua anh dich
		uchar * psRow = psData;//Con tro dong cua anh nguon
		for (int j = 0; j < width; j++, pRow += nChannels, psRow += nChannels) {
			for (int k = 0; k < nChannels; k++)
				pRow[k] = lookup[(int)psRow[k]];
		}
	}
	if (destinationImage.empty())
		return false; //Loi 
	return true;
}

// Ham thay doi do sang cua anh
// Input: anh nguon, anh dich, tham so
// Output: thuc hien duoc hay khong? (true/ false)
bool ChangeContrast(const Mat & sourceImage, Mat & destinationImage, float c)
{
	if (sourceImage.empty())
		return 0;
	//Tạo bảng lookup
	uchar lookup[256];
	for (int i = 0; i < 256; i++)
		lookup[i] = saturate_cast<uchar>(i * c);

	//Khởi tạo ảnh đích có kích thước và type giống ảnh nguồn
	destinationImage.create(sourceImage.rows, sourceImage.cols, sourceImage.type());

	int width = sourceImage.cols, height = sourceImage.rows;
	//nChannels là số kênh màu
	int nChannels = sourceImage.channels();
	//widthStep là khoảng cách tính theo byte giữa 2 pixel cùng cột trên 2 dòng kế tiếp
	int widthStep = sourceImage.step[0];
	//pData là con trỏ quản lý vùng nhớ ảnh
	uchar* pData = (uchar*)destinationImage.data; //Con tro data cua anh dich
	uchar* psData = (uchar*)sourceImage.data; //Con tro data cua anh nguon
	for (int i = 0; i < height; i++, psData += widthStep, pData += widthStep) {
		uchar * pRow = pData; //Con tro dong cua anh dich
		uchar * psRow = psData;//Con tro dong cua anh nguon
		for (int j = 0; j < width; j++, pRow += nChannels, psRow += nChannels) {
			for (int k = 0; k < nChannels; k++)
				pRow[k] = lookup[(int)psRow[k]];
		}
	}
	if (destinationImage.empty())
		return false; //Loi 
	return true;
}