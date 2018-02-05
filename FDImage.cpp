
#pragma   comment(lib,   "vfw32.lib ")
#pragma comment (lib , "comctl32.lib")

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// 新版本写在下面文件中：
#include <opencv2/nonfree/features2d.hpp>
//#include "opencv2/features2d/features2d.hpp"
#include<opencv2/legacy/legacy.hpp>
#include <algorithm>
#include <fstream>
using namespace std;

#include "FDImage.h"
#include "CCCascadeClassifier.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

FDImage::FDImage():height(0),width(0),data(NULL),label(-1),variance(0.0)
{
	for(int i=0;i<height;i++){
			data[i] = new REAL;
		}
		buf = new REAL;

}

FDImage::~FDImage()
{
	delete [] img;
	Clear();
}

void FDImage::Clear(void)
{

	if(data == NULL)
		assert(buf == NULL);
	else
	{
		assert(buf != NULL);
		for(int i=0;i<height;i++){
			data[i] = NULL;
		}
		delete[] data;	data = NULL;
		delete[] buf;  	buf = NULL;
	
		height = width = 0;
		variance = 0.0;
		label = -1;
	}	

}

void FDImage::cleartmp()
{
	if(data == NULL)
		assert(buf == NULL);
	else
	{
		assert(buf != NULL);
		for(int i=0;i<height;i++){
			data[i] = NULL;
		}
		delete[] data;	data = NULL;
		delete[] buf;  	buf = NULL;
	}
	//if(data  != NULL){
	//	for(int i=0;i<height;i++)	data[i] = NULL;
	//	delete[] data;	data = NULL;
	//	delete[] buf;	buf = NULL;
	//}
	//if(m_gradientImage != NULL){
	//	delete [] m_gradientImage;
	//	m_gradientImage=NULL;
	//}
}
void FDImage::Copy(const FDImage& source)
	// the ONLY way to make a copy of 'source' to this image
{
	assert(source.height > 0);
	assert(source.width > 0);
	if(&source == this)	return;
	SetSize(cvSize(source.height,source.width));
	label = source.label;
	memcpy(buf,source.buf,sizeof(REAL)*height*width);
}

void FDImage::Load(IplImage* temp)
{
	img = temp;
	SetSize(cvSize(img->height,img->width));
	for(int i=0,ih=img->height,iw=img->width;i<ih;i++)
	{
		REAL* pdata = data[i];
		unsigned char* pimg = reinterpret_cast<unsigned char*>(img->imageData+img->widthStep*i);
		for(int j=0;j<iw;j++) pdata[j] = pimg[j];
	}
}

void FDImage::SetSize(const CvSize size)
	// 'size' is the new size of the image, if necessary, memory is reallocated
	// size.cx is the new height and size.cy is the new width
{
	if((size.height == height) && (size.width == width) && (buf != NULL) &&(data != NULL) ) return; 
	assert(size.height >= 0); assert(size.width >= 0);
	//Clear();
	height = size.height;	width = size.width;
	buf = new REAL[height*width]; assert(buf != NULL);
	data = new REAL*[height];	assert(data != NULL);
	for(int i=0;i<height;i++)	data[i] = &buf[i*width];

}

FDImage& FDImage::operator=(const FDImage& source)
{
	SetSize(cvSize(source.height,source.width));
	//memcpy(data,source.data,sizeof(double)*height*width);
	label = source.label;

	return *this;
}

void FDImage::Resize(FDImage &result, REAL ratio) const
{
	result.SetSize(cvSize(int(height*ratio),int(width*ratio)));
	ratio = 1/ratio;
	for(int i=0,rh=result.height,rw=result.width;i<rh;i++)
		for(int j=0;j<rw;j++) {
			int x0,y0;
			REAL x,y,fx0,fx1;
			x = j*ratio; y = i*ratio;
			x0 = (int)(x);
			y0 = (int)(y);

			//by Jianxin Wu  
			//1. The conversion of float to int in C is towards to 0 point, i.e. the floor function for positive numbers, and ceiling function for negative numbers.
			//2. We only make use of ratio<1 in this applicaiton, and all numbers involved are positive.
			//Using these, we have 0<=x<=height-1 and 0<=y<=width-1. Thus, boundary conditions check is not necessary.
			//In languages other than C/C++ or ratio>=1, take care. 
			if (x0 == width-1) x0--;
			if (y0 == height-1) y0--;

			x = x - x0; y = y - y0;

			fx0 = data[y0][x0] + x*(data[y0][x0+1]-data[y0][x0]);
			fx1 = data[y0+1][x0] + x*(data[y0+1][x0+1]-data[y0+1][x0]);

			result.data[i][j] = fx0 + y*(fx1-fx0);
		}
}

void FDImage::CutToLxL(FDImage &result, CvRect rect, int iL) const
{
	CvSize dst_cvsize; IplImage* pdst = 0;
	dst_cvsize.width = iL; //ƒø±ÍÕºœÒøÌ∂»  
	dst_cvsize.height = iL; //ƒø±ÍÕºœÒ∏ﬂ∂» 
	pdst = cvCreateImage(dst_cvsize, IPL_DEPTH_8U, 1);
	
	// Ω´ƒ⁄¥Ê÷–µƒÕº∆¨ –¥µΩ opencv Ω·ππµƒÕº∆¨÷–
	IplImage* tsimg;
	tsimg = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
	for(int i=0,ih=tsimg->height,iw=tsimg->width;i<ih;i++)
	{
		REAL* pdata = data[i];
		unsigned char* pimg = reinterpret_cast<unsigned char*>(tsimg->imageData+tsimg->widthStep*i);
		for(int j=0;j<iw;j++) pimg[j] = (unsigned char)pdata[j];
	}
	// ¿˚”√ opencv ∫Ø ˝Ω¯––Àı∑≈
	cv::Mat img(tsimg);
	cv::Mat temp;
	img(rect).copyTo(temp);
	IplImage *src = cvCreateImage(cvSize(temp.cols,temp.rows),8,1);
	
	cvResize(src,pdst,CV_INTER_LINEAR);
	// ‘⁄øΩµΩƒø±Í∂‘œÛ÷–

	result.SetSize(cvSize(pdst->height,pdst->width));
	for(int i=0,ih=pdst->height,iw=pdst->width;i<ih;i++)
	{
		REAL* pdata = result.data[i];
		unsigned char* pimg = reinterpret_cast<unsigned char*>(pdst->imageData+pdst->widthStep*i);
		for(int j=0;j<iw;j++) pdata[j] = pimg[j];
	}

	cvReleaseImage(&pdst);cvReleaseImage(&tsimg);

}

void FDImage::Resize240x40(FDImage &result) const
{
	CvSize dst_cvsize; IplImage* pdst = 0;
	dst_cvsize.width = 40; //ƒø±ÍÕºœÒøÌ∂»  
	dst_cvsize.height = 40; //ƒø±ÍÕºœÒ∏ﬂ∂» 
	pdst = cvCreateImage(dst_cvsize, IPL_DEPTH_8U, 1);
	
	// Ω´ƒ⁄¥Ê÷–µƒÕº∆¨ –¥µΩ opencv Ω·ππµƒÕº∆¨÷–
	IplImage* tsimg;
	tsimg = cvCreateImage(cvSize(width,height),IPL_DEPTH_8U,1);
	for(int i=0,ih=tsimg->height,iw=tsimg->width;i<ih;i++)
	{
		REAL* pdata = data[i];
		unsigned char* pimg = reinterpret_cast<unsigned char*>(tsimg->imageData+tsimg->widthStep*i);
		for(int j=0;j<iw;j++) pimg[j] = (unsigned char)pdata[j];
	}
	// ¿˚”√ opencv ∫Ø ˝Ω¯––Àı∑≈
	cvResize(tsimg,pdst,CV_INTER_LINEAR);
	// ‘⁄øΩµΩƒø±Í∂‘œÛ÷–

	result.SetSize(cvSize(pdst->height,pdst->width));
	for(int i=0,ih=pdst->height,iw=pdst->width;i<ih;i++)
	{
		REAL* pdata = result.data[i];
		unsigned char* pimg = reinterpret_cast<unsigned char*>(pdst->imageData+pdst->widthStep*i);
		for(int j=0;j<iw;j++) pdata[j] = pimg[j];
	}

	cvReleaseImage(&pdst);cvReleaseImage(&tsimg);

}
