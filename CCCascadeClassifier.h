
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// 新版本写在下面文件中：
#include <opencv2/nonfree/features2d.hpp>
//#include "opencv2/features2d/features2d.hpp"
#include<opencv2/legacy/legacy.hpp>

#include <vector>
using namespace std;
using namespace cv;

class FDImage;

struct rectScore{
	Rect rect;
	float score;
};


struct CCCascadeClassifier
{
	int count;
	int indexBoosted;
	CCCascadeClassifier();
	~CCCascadeClassifier();
	void Clear(void);
	CCCascadeClassifier& operator=(const CCCascadeClassifier& source);

	vector<Rect>  FaceDetectWithRet_ScaleIamge(FDImage& original,const string filename);
	
    void DrawResults(FDImage& image, vector<Rect>& results) const;
	void DrawResults2(FDImage& image, const vector<rectScore>& results) const;
    void PostProcess(vector<Rect>& result, const int combine_min);
    void nms(
             const std::vector<cv::Rect>& srcRects,
             std::vector<cv::Rect>& resRects,
             float thresh
             );



	double Expression_Recognition(FDImage& original,const string filename);

	void getInputData(FDImage &im,double* data,int x,int y,int rh,int rw, REAL ratio);

	FDImage procface;
	double value;
	double data40[1600];
	double data24[576];
	double data20[400];
	double T_time;
	Rect rect;
	REAL ratio;
	vector<Rect> results;
    vector<Rect> res;
	FDImage image;
	Rect rect40;
};

