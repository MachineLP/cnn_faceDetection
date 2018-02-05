#include "stdafx.h"
#include <fstream>
#include <vector>
#include <math.h>
#include <string>
#include <algorithm>
#include <set>
using namespace std;

#include "FDImage.h"
#include "CCCascadeClassifier.h"

#include "mlp.h"  
#include "util.h"  
// #include "testinherit.h"  
#include "neuralNetwork.h" 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// 新版本写在下面文件中：
#include <opencv2/nonfree/features2d.hpp>
//#include "opencv2/features2d/features2d.hpp"
#include<opencv2/legacy/legacy.hpp>

using namespace cv;

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

CCCascadeClassifier::CCCascadeClassifier():count(0),indexBoosted(0)
{
    // resultsGroup.resize(9);
}

CCCascadeClassifier::~CCCascadeClassifier()
{
    Clear();
}

void CCCascadeClassifier::Clear()
{
    count = 0;
}

CCCascadeClassifier& CCCascadeClassifier::operator=(const CCCascadeClassifier& source)
{
    Clear();
    count = source.count;
    return *this;
}

void CCCascadeClassifier::DrawResults(FDImage& image, vector<Rect>& results) const
{
	int i;
	unsigned int k;
	int x1,x2,y1,y2;

	for(k=0;k<results.size();k++)
	{
		y1 = (results[k].y>=0)?results[k].y:0;
		y1 = (results[k].y<image.height)?results[k].y:(image.height-1);
		y2 = (results[k].y+results[k].height>=0)?results[k].y+results[k].height:0;
		y2 = (results[k].y+results[k].height<image.height)?results[k].y+results[k].height:(image.height-1);
		x1 = (results[k].x>=0)?results[k].x:0;
		x1 = (results[k].x<image.width)?results[k].x:(image.width-1);
		x2 = (results[k].x+results[k].width>=0)?results[k].x+results[k].width:0;
		x2 = (results[k].x+results[k].width<image.width)?results[k].x+results[k].width:(image.width-1);
		for(i=y1;i<=y2;i++) 
		{
			image.data[i][x1] = 255;
			image.data[i][x2] = 255;
		}
		for(i=x1;i<=x2;i++)
		{
			image.data[y1][i] = 255;
			image.data[y2][i] = 255;
		}
		results[k].x = x1;
		results[k].y = y1;
		results[k].width = x2 - results[k].x;
		results[k].height = y2 - results[k].y;
	}
}

void CCCascadeClassifier::DrawResults2(FDImage& image,const vector<rectScore>& results) const
{
	int i;
	unsigned int k;
	int x1,x2,y1,y2;

	for(k=0;k<results.size();k++)
	{
		y1 = (results[k].rect.y>=0)?results[k].rect.y:0;
		y1 = (results[k].rect.y<image.height)?results[k].rect.y:(image.height-1);
		y2 = (results[k].rect.y+results[k].rect.height>=0)?results[k].rect.y+results[k].rect.height:0;
		y2 = (results[k].rect.y+results[k].rect.height<image.height)?results[k].rect.y+results[k].rect.height:(image.height-1);
		x1 = (results[k].rect.x>=0)?results[k].rect.x:0;
		x1 = (results[k].rect.x<image.width)?results[k].rect.x:(image.width-1);
		x2 = (results[k].rect.x+results[k].rect.width>=0)?results[k].rect.x+results[k].rect.width:0;
		x2 = (results[k].rect.x+results[k].rect.width<image.width)?results[k].rect.x+results[k].rect.width:(image.width-1);
		// ’‚µÿ∑ΩæÕ «ª≠≥ˆ¿¥»À¡≥øÚ£¨∞—»À¡≥µƒŒª÷√¥Ú…œ∞◊µ„
		for(i=y1;i<=y2;i++) 
		{
			image.data[i][x1] = 255;
			image.data[i][x2] = 255;
		}
		for(i=x1;i<=x2;i++)
		{
			image.data[y1][i] = 255;
			image.data[y2][i] = 255;
		}
	}
}


// ¥À¥¶µƒx,y±Ì æµƒ «x––y¡–µƒ÷µ,∂¯≤ª «◊¯±Í£¨◊™ªØŒ™openCV÷–µƒ◊¯±ÍŒ™£®y£¨x£©
void CCCascadeClassifier::getInputData(FDImage &im,double* data,int x,int y,int rh,int rw, REAL wRatio)
{
	if (wRatio == 1)
	{
		for(int i=0; i<rh; i++)
		{
			for(int j=0; j<rw; j++)
			{
				data[i*rw+j] = im.data[i+x][j+y]/256.0;
			}
		}
	}
	else
	{
		FDImage image;
		Rect rect;
		// openCV÷–µƒ◊¯±Í◊™ªØ°£
		rect.x = y;
		rect.y = x;
		rect.height = rh*wRatio;
		rect.width = rw*wRatio;
		im.CutToLxL(image,rect,40);
		for(int i=0; i<rh/wRatio; i++)
		{
			for(int j=0; j<rw/wRatio; j++)
			{
				data[i*rw+j] = image.data[i][j]/256.0;
			}
		}
	}
}

vector<Rect> CCCascadeClassifier::FaceDetectWithRet_ScaleIamge(FDImage& original,const string filename)
{
	int sx = 40;
	int sy = 40;

	ratio = 1.0;
	original.Resize(procface,ratio);
	results.clear();
    
    // 步长
    // int stripSize = sx/5;
    int stripSize = 8;
    
	while((procface.height+1>sx+1) && (procface.width+1>sy+1))
	{
		for(int i=0,size_x=procface.height+1-sx;i<size_x;i+=stripSize)
			for(int j=0,size_y=procface.width+1-sy;j<size_y;j+=stripSize){
				
				getInputData(procface,data40,i,j,sx,sy,1);

				// 40*40 CNNºÏ≤‚¥∞ø⁄
				int iResult40 = TestCnnTheano(sx, 2, data40, T_time);

				if (iResult40 == 1)
				{
                    /*
				    int iWidth24 = 24;
					// openCV÷–µƒ◊¯±Í◊™ªØ°£
					rect40.x = j;
					rect40.y = i;
					rect40.height = sx;
					rect40.width = sy;
					procface.CutToLxL(image,rect40,iWidth24);
					for(int m=0; m<iWidth24; m++)
					{
						for(int n=0; n<iWidth24; n++)
						{
							data24[m*iWidth24+n] = image.data[m][n]/256.0;
						}
					}
					// 24*24 CNNºÏ≤‚¥∞ø⁄
					int iResult24 = TestCnnTheano24x24(iWidth24, 2, data24, T_time);

					if (iResult24 == 1)
					{*/
                        /*
						//FDImage image;
						//Rect rect40;
						int iWidth20 = 20;
						// openCV÷–µƒ◊¯±Í◊™ªØ°£
						rect40.x = j;
						rect40.y = i;
						rect40.height = sx;
						rect40.width = sy;
						procface.CutToLxL(image,rect40,iWidth20);
						for(int i=0; i<iWidth20; i++)
						{
							for(int j=0; j<iWidth20; j++)
							{
								data20[i*iWidth20+j] = image.data[i][j]/256.0;
							}
						}
						// 20*20 CNNºÏ≤‚¥∞ø⁄
						int iResult20 = TestCnnTheano20x20(iWidth20, 2, data20, T_time);

						if (iResult20 == 1)
                         */

						{
                            
							const REAL r = 1.0/ratio;
							rect.x = (long)(j*r);rect.y = (long)(i*r);
							rect.width = (long)((j+sx)*r)-rect.x;rect.height = (long)((i+sy)*r)-rect.y;
							results.push_back(rect);
						}
					//}
				}
			}
            // 0.8, 0.6
			ratio = ratio*0.8;
			procface.Clear();
			original.Resize(procface,ratio);
	}

    if(results.size()){
	    // PostProcess(results,100);
	    // PostProcess(results,4);
        
        PostProcess(results,5);
	    //PostProcess(results,0);
        
	    // DrawResults(original,results);
	    // original.Save(filename+"_result.JPG");
        
        //nms(results,res,0.6f);
        //results = res;
        
	    
    }
    return results;
}

inline int SizeOfRect(const Rect& rect)
{
	return rect.height*rect.width;
}
// PostProcess的参数：所有检测窗口 和 combine_min值。
void CCCascadeClassifier::PostProcess(vector<Rect>& result,const int combine_min)
{
	vector<Rect> res1;
	vector<Rect> resmax;
	vector<int> res2;
	bool yet;
	Rect rectInter,rectUnion;

	for(unsigned int i=0,size_i=result.size();i<size_i;i++){
		yet = false;
		Rect& result_i = result[i];
		for(unsigned int j=0,size_r=res1.size();j<size_r;j++){
			Rect& resmax_j = resmax[j];
            rectInter = result_i & resmax_j;
            rectUnion = result_i | resmax_j;
			if( rectInter.area() != 0 && rectUnion.area() != result_i.area()){
			//	if((double)SizeOfRect(rectInter)/SizeOfRect(rectUnion) > 0.6){
				if((double)SizeOfRect(rectInter)/SizeOfRect(result_i) > 0.6 &&
					(double)SizeOfRect(rectInter)/SizeOfRect(resmax_j) > 0.6 ){
					Rect& res1_j = res1[j];
					resmax_j= resmax_j | result_i;
                    //int temp = res1_j.y;
                    res1_j.y += result_i.y;
                    //res1_j.height = res1_j.height + temp + result_i.height + result_i.y - res1_j.y;
                    res1_j.height += result_i.height;
                    //int tmp = res1_j.x;
					res1_j.x += result_i.x;
                    //res1_j.width = res1_j.width + tmp + result_i.width + result_i.x - res1_j.x;
					res1_j.width += result_i.width;
					res2[j]++;
					yet = true;
					break;
				}
			}
		}
		if(yet==false){
			res1.push_back(result_i);
			resmax.push_back(result_i);
			res2.push_back(1);
		}
	}

	for(unsigned int i=0,size=res1.size();i<size;i++){
		const int count = res2[i];
		Rect& res1_i = res1[i];
        //int tmp1 = res1_i.y;
		res1_i.y /= count;
        //res1_i.height = (res1_i.height + tmp1)/count - res1_i.y;
		res1_i.height /= count;
        //int tmp2 = res1_i.x;
		res1_i.x /= count;
        //res1_i.width = (res1_i.width + tmp2)/count - res1_i.x;
		res1_i.width /= count;
	}

	vector<Rect> result1,result2;vector<int> res3,res4;
	for(int i=0,size=res1.size();i<size;i++){
		if(res2[i]>combine_min){
			result2.push_back(res1[i]);
			res4.push_back(res2[i]);
		}
	}
    
	bool bcan;
	while(true){
		bcan = false;
		vector<int> toDel;
		result1.clear();result1 = result2;
		res3.clear();res3 = res4;
		for(int i=0,size_i=result1.size();i<size_i;i++)
			toDel.push_back(1);

		for(int i=0,size_i=result1.size();i<size_i;i++)
		{
			Rect& result_i = result1[i];
			for(unsigned int j=i+1,size_j=result1.size();j<size_j;j++)
			{
				Rect& result_j = result1[j];
                rectInter = result_i & result_j;
				if(rectInter.area() != 0)
				{
					double areaInter = (double)SizeOfRect(rectInter),areaI = (double)SizeOfRect(result_i),areaJ=(double)SizeOfRect(result_j);
					//double s1 = (double)(areaInter/areaI)/(areaInter/areaJ);
					//double s2 = (double)(areaInter/areaJ)/(areaInter/areaI);
					double s1 = areaInter/areaI;
					double s2 = areaInter/areaJ;
					if( s1/s2 >  1 || s2/s1 > 1 || areaInter/areaI > 0.6 || areaInter/areaJ > 0.6){
						if(res3[i] > res3[j])
							toDel[j] = 0;
						else
							toDel[i] = 0;
						bcan = true;
					}
				}
			}
		}
		if(bcan == false)
			break;
		result2.clear();res4.clear();
		for(unsigned int i=0,size=result1.size();i<size;i++) 
			if(toDel[i] == 1){
				result2.push_back(result1[i]);
				res4.push_back(res3[i]);
			}
	}
    
	result.clear();
	result = result2;
}

void CCCascadeClassifier::nms(
         const std::vector<cv::Rect>& srcRects,
         std::vector<cv::Rect>& resRects,
         float thresh
         )
{
    resRects.clear();
    
    const size_t size = srcRects.size();
    if (!size)
    {
        return;
    }
    
    // Sort the bounding boxes by the bottom - right y - coordinate of the bounding box
    std::multimap<int, size_t> idxs;
    for (size_t i = 0; i < size; ++i)
    {
        idxs.insert(std::pair<int, size_t>(srcRects[i].br().y, i));
    }
    
    // keep looping while some indexes still remain in the indexes list
    while (idxs.size() > 0)
    {
        // grab the last rectangle
        auto lastElem = --std::end(idxs);
        const cv::Rect& rect1 = srcRects[lastElem->second];
        
        resRects.push_back(rect1);
        
        idxs.erase(lastElem);
        
        for (auto pos = std::begin(idxs); pos != std::end(idxs); )
        {
            // grab the current rectangle
            const cv::Rect& rect2 = srcRects[pos->second];
            
            float intArea = (rect1 & rect2).area();
            float unionArea = rect1.area() + rect2.area() - intArea;
            float overlap = intArea / unionArea;
            
            // if there is sufficient overlap, suppress the current bounding box
            if (overlap > thresh)
            {
                pos = idxs.erase(pos);
            }
            else
            {
                ++pos;
            }
        }
    }
}
