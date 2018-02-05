//
//  main.cpp
//  faceDetection-cnn
//
//  Created by 刘鹏 on 2016/11/27.
//  Copyright © 2016年 刘鹏. All rights reserved.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
// 新版本写在下面文件中：
#include <opencv2/nonfree/features2d.hpp>
//#include "opencv2/features2d/features2d.hpp"
#include<opencv2/legacy/legacy.hpp>

#include <iostream>
#include <stdio.h>

#include "FDImage.h"
#include "CCCascadeClassifier.h"
#include <time.h>
#include <stdio.h>
#include <algorithm>

using namespace std;
using namespace cv;


IplImage* cutImage(IplImage* src, CvRect rect) {
    cvSetImageROI(src, rect);
    IplImage* dst = cvCreateImage(cvSize(rect.width, rect.height),
                                  src->depth,
                                  src->nChannels);
    
    cvCopy(src,dst,0);
    cvResetImageROI(src);
    // imshow("dst",(Mat)dst);
    return dst;
}

// Function main
int main()
{
    
    FDImage* image;
    image = new FDImage;
    vector<Rect> vFaceRect;
    vector<Rect> faceBox;
    CCCascadeClassifier *cascade;
    cascade = new CCCascadeClassifier;
    //========================================================
    // CvCapture  «“ª∏ˆΩ·ππÃÂ£¨”√¿¥±£¥ÊÕºœÒ≤∂ªÒÀ˘–Ë“™µƒ–≈œ¢°£
    // opencvÃ·π©¡Ω÷÷∑Ω Ω¥”Õ‚≤ø≤∂ªÒÕºœÒ£¨“ª÷÷ «¥”…„œÒÕ∑÷–£¨“ª÷÷
    //  «Õ®π˝Ω‚¬Î ”∆µµ√µΩÕºœÒ°£¡Ω÷÷∑Ω Ω∂º±ÿ–Î¥”µ⁄“ª÷°ø™ º“ª÷°“ª÷°
    // µƒ∞¥À≥–ÚªÒ»°£¨“Ú¥À√øªÒ»°“ª÷°∫Û∂º“™±£¥Êœ‡”¶µƒ◊¥Ã¨∫Õ≤Œ ˝°£
    // ±»»Á¥” ”∆µŒƒº˛÷–ªÒ»°£¨–Ë“™±£¥Ê ”∆µŒƒº˛µƒŒƒº˛√˚£¨œ‡”¶µƒ******
    // ¿‡–Õ£¨œ¬“ª¥Œ»Áπ˚“™ªÒ»°Ω´–Ë“™Ω‚¬Îƒƒ“ª÷°µ»°£ ’‚–©–≈œ¢∂º±£¥Ê‘⁄
    // CvCaptureΩ·ππ÷–£¨√øªÒ»°“ª÷°∫Û£¨’‚–©–≈œ¢∂ºΩ´±ª∏¸–¬£¨ªÒ»°œ¬“ª÷°
    // –Ë“™Ω´–¬–≈œ¢¥´∏¯ªÒ»°µƒapiΩ”ø⁄
    //=======================================================
    CvCapture* capture = 0;
    //===========================================================
    // IplImage  «Ω·ππÃÂ¿‡–Õ£¨”√¿¥±£¥Ê“ª÷°ÕºœÒµƒ–≈œ¢£¨“≤æÕ «“ª÷°
    // ÕºœÒµƒÀ˘”–œÒÀÿ÷µππ≥…µƒ“ª∏ˆæÿ’Û
    //===========================================================
    IplImage *frame, *frame_copy = 0;
    IplImage *src;
    IplImage *pGrayImg;
    
    // ¥¥Ω®“ª∏ˆ¥∞ø⁄£¨”√°∞result°±◊˜Œ™¥∞ø⁄µƒ±Í ∂∑˚
    cvNamedWindow( "result", 1 );
    
    // ==========================================
    // ≥ı ºªØ“ª∏ˆ ”∆µ≤∂ªÒ≤Ÿ◊˜°£
    // ∏ÊÀﬂµ◊≤„µƒ≤∂ªÒapiŒ“œÎ¥” Capture1.avi÷–≤∂ªÒÕº∆¨£¨
    // µ◊≤„apiΩ´ºÏ≤‚≤¢—°‘Òœ‡”¶µƒ******≤¢◊ˆ∫√◊º±∏π§◊˜
    //==============================================
    capture = cvCaptureFromFile("/Users/liupeng/Desktop/my/faceDetection-cnn/faceDetection-cnn/data/outColor2.mp4");
    // ªÒ»°…„œÒÕ∑
    // capture = cvCaptureFromCAM(0);
    
    // »Áπ˚ ≥ı ºªØ ß∞‹£¨ƒ«√¥captureŒ™ø’÷∏’Î£¨≥Ã–ÚÕ£÷π£¨
    // ∑Ò‘ÚΩ¯»Î≤∂ªÒ—≠ª∑
    if( capture )
    {
        // ≤∂ªÒ—≠ª∑
        while(1)
        {
            // µ˜”√cvGrabFrame,»√µ◊≤„apiΩ‚¬Î“ª÷°ÕºœÒ
            // »Áπ˚Ω‚¬Î ß∞‹£¨æÕÕÀ≥ˆ—≠ª∑
            // »Áπ˚≥…π¶£¨Ω‚¬ÎµƒÕºœÒ±£¥Ê‘⁄µ◊≤„apiµƒª∫¥Ê÷–
            if( !cvGrabFrame( capture ))
                break;
            
            // Ω´Ω‚¬Îµ√µΩÕºœÒ–≈œ¢¥”ª∫¥Ê÷–◊™ªª≥…IplImage∏Ò Ω∑≈‘⁄frame÷–
            // frame = cvRetrieveFrame( capture );
            frame = cvQueryFrame(capture);
            
            // »Áπ˚ªÒ»°ª∫¥ÊªÚ◊™ªª ß∞‹£¨‘ÚÕÀ≥ˆ—≠ª∑
            if( !frame )
                break;
            
            Mat mFrame = (Mat)frame;
            // Apply the classifier to the frame
            if (!mFrame.empty()){
                
                src = frame;
                cvShowImage( "image", src );
                
                CvRect rect;
                int iX = 900;
                int iY = 200;
                
                rect.x = 200;
                rect.y = 0;
                rect.width = 200;
                rect.height = 200;
                
                /* Infrared ÕºœÒœ¬µƒ…Ë÷√
                 rect.x = 12;
                 rect.y = 24;
                 rect.width = 400;
                 rect.height = 400; */
                
                IplImage *temp = cutImage(src,rect);
                // imshow("aaaa",Mat(temp));
                
                // µ˜’˚Õº∆¨µƒ¥Û–°°£
                IplImage *imageresize;
                imageresize=cvCreateImage(cvSize(200,200),temp->depth,temp->nChannels);
                cvResize(temp,imageresize,CV_INTER_LINEAR);
                // imshow("img1", Mat(temp));
                // waitKey();
                
                pGrayImg=cvCreateImage(cvGetSize(imageresize),8,1);
                cvCvtColor(imageresize,pGrayImg,CV_RGB2GRAY);
                
                imshow("img2",Mat(pGrayImg));
                // waitKey();
                
                // Ω´IplImage∏Ò ΩµƒÕº∆¨◊™ªØŒ™FDImage∏Ò ΩµƒÕº∆¨
                image->Load(pGrayImg);
                double t1, t2;
                //cascade->LoadDefaultCascade();
                //if(cascade ==NULL)
                ///	return;
                FDImage &im = *image;
                //if(cascade->count>0)
                {
                    t1 = getTickCount();
                    //cascade->FaceDetect_ScaleImage(im);
                    vFaceRect = cascade->FaceDetectWithRet_ScaleIamge(im,"E:/");
                    for (int ic = 0; ic < vFaceRect.size(); ic++) // Iterate through all current elements (detected faces)
                    {
                        if(vFaceRect[ic].x > 0)
                        {
                            float alpha = 1.0;
                            Point pt1(200+vFaceRect[ic].x*alpha, vFaceRect[ic].y*alpha); // Display detected faces on main window - live stream from camera
                            Point pt2((200+vFaceRect[ic].x+vFaceRect[ic].width)*alpha, (vFaceRect[ic].y+vFaceRect[ic].height)*alpha);
                            Mat mF = (Mat)frame;
                            rectangle(mF, pt1, pt2, Scalar(0, 255, 0), 2, 8, 0);
                        }
                    }
                }
            }
            else{
                printf(" --(!) No captured frame -- Break!");
                //break;
            }
            
            int c = waitKey(1);
            
            if (27 == char(c)){
                // break;
            }
            
            // Ω´frame÷–µƒÕºœÒ–≈œ¢‘⁄¥∞ø⁄result÷–œ‘ æ
            cvShowImage( "gray", pGrayImg );
            cvShowImage( "result", frame );
            
            // ‘›Õ£“ªª·∂˘£¨»√ƒ„ø¥“ªœ¬ÕºœÒ
            cvWaitKey(1);
            
        }
        
        // ÕÀ≥ˆ≥Ã–Ú÷Æ«∞“™«Â¿Ì“ªœ¬∂—’ª÷–µƒƒ⁄¥Ê£¨√‚µ√ƒ⁄¥Ê–π¬∂
        //cvReleaseImage( &frame );◊¢“‚≤ª–Ë“™’‚æ‰£¨“ÚŒ™frame «¥” ”∆µ÷–≤∂ªÒµƒ£¨√ª”–µ•∂¿∑÷≈‰ƒ⁄¥Ê£¨Œﬁ–Ë Õ∑≈£¨µ±capture  Õ∑≈µƒ ±∫Úframe◊‘»ªæÕ Õ∑≈¡À°£
        
        // ÕÀ≥ˆ÷Æ«∞Ω· ¯µ◊≤„apiµƒ≤∂ªÒ≤Ÿ◊˜£¨√‚µ√À¸√«’º◊≈√©ø”≤ª¿≠ ∫
        // ±»»Áª· πµ√±µƒ≥Ã–ÚŒﬁ∑®∑√Œ “—æ≠±ªÀ¸√«¥Úø™µƒŒƒº˛
        cvReleaseCapture( &capture );
        
    }
    cvDestroyWindow("result");
    
    return 0;
}



