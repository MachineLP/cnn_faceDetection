#define USE_DOUBLE

#ifdef USE_DOUBLE
typedef double REAL;
#else
typedef float REAL;
#endif


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// 新版本写在下面文件中：
#include <opencv2/nonfree/features2d.hpp>
//#include "opencv2/features2d/features2d.hpp"
#include<opencv2/legacy/legacy.hpp>

class FDImage
{
public:
	FDImage();
	~FDImage();

	void Clear(void); 
	void SetSize(const CvSize size);
	FDImage& operator=(const FDImage& source);
	void Resize(FDImage &result,  REAL ratio) const;
	void Resize240x40(FDImage &result) const;
	void CutToLxL(FDImage &result, CvRect rect, int iL) const;
	void Copy(const FDImage& source);
	void Load(IplImage* img);
	void cleartmp();

public:
	int height; // height, or, number of rows of the image
	int width;  // width, or, number of columns of the image
	REAL** data;  // auxiliary pointers to accelerate the read/write of the image
	// no memory is really allocated, use memory in (buf)
	// data[i][j] is a pixel's gray value in (i)th row and (j)th column
	REAL* buf;    // pointer to a block of continuous memory containing the image
	int label;
	REAL variance;

//	SIFT
	//Mat 

	IplImage* img;
};
