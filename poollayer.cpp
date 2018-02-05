#include "stdafx.h"
#include "poollayer.h"  
#include "util.h"  
#include <cassert>  
  
PoolLayer::PoolLayer(int iOutImageNumber, int iPoolWidth, int iFeatureMapWidth):  
    m_iOutImageNumber(iOutImageNumber),  
    m_iPoolWidth(iPoolWidth),  
    m_iFeatureMapWidth(iFeatureMapWidth)  
{  
    m_iPoolSize = m_iPoolWidth * m_iPoolWidth;  
    m_iOutImageEdge = m_iFeatureMapWidth / m_iPoolWidth;  
    m_iOutImageSize = m_iOutImageEdge * m_iOutImageEdge;  
  
    m_pdOutData = new double[m_iOutImageNumber * m_iOutImageSize];  
    m_pdBias = new double[m_iOutImageNumber];  
    /*for (int i = 0; i < m_iOutImageNumber; ++i) 
    { 
        m_pdBias[i] = 1; 
    }*/  
      
  
}  
  
PoolLayer::~PoolLayer()  
{  
    delete []m_pdOutData;  
    delete []m_pdBias;  
}  
  
void PoolLayer::Convolute(double *pdInputData)  
{  
    int iFeatureMapSize = m_iFeatureMapWidth * m_iFeatureMapWidth;  
    for (int iOutImageIndex = 0; iOutImageIndex < m_iOutImageNumber; ++iOutImageIndex)  
    {  
        double dBias = m_pdBias[iOutImageIndex];  
        for (int i = 0; i < m_iOutImageEdge; ++i)  
        {  
            for (int j = 0; j < m_iOutImageEdge; ++j)  
            {  
                double dValue = 0.0;  
                int iInputIndex, iInputIndexStart, iOutIndex;  
                /************************************************************************/  
                /* 这里是最大的bug，dMaxPixel初始值设置为0，然后找最大值 
                ** 问题在于像素值有负数，导致后面一系列计算错误，实在是太难找了 
                /************************************************************************/  
                double dMaxPixel = INT_MIN ;  
                iOutIndex = iOutImageIndex * m_iOutImageSize + i * m_iOutImageEdge + j;  
                iInputIndexStart = iOutImageIndex * iFeatureMapSize + (i * m_iFeatureMapWidth + j) * m_iPoolWidth;  
                for (int m = 0; m < m_iPoolWidth; ++m)  
                {  
                    for (int n = 0; n < m_iPoolWidth; ++n)  
                    {  
                        //                  int iPoolIndex = m * m_iPoolWidth + n;  
                        //i am not sure, the expression of below is correct?  
                        iInputIndex = iInputIndexStart + m * m_iFeatureMapWidth + n;  
                        if (pdInputData[iInputIndex] > dMaxPixel)  
                        {  
                            dMaxPixel = pdInputData[iInputIndex];  
                        }  
                    }//end n  
                }//end m  
                dValue = dMaxPixel + dBias;  
  
                assert(iOutIndex < m_iOutImageNumber * m_iOutImageSize);  
                //m_pdOutData[iOutIndex] = (dMaxPixel);  
                m_pdOutData[iOutIndex] = mytanh(dValue);  
            }//end j  
        }//end i  
    }//end iOutImageIndex  
}  
  
void PoolLayer::SetBias(const vector<double> &vecBias)  
{  
    assert(vecBias.size() == (DWORD)m_iOutImageNumber);  
    for (int i = 0; i < m_iOutImageNumber; ++i)  
    {  
        m_pdBias[i] = vecBias[i];  
    }  
}  
  
double* PoolLayer::GetOutputData()  
{  
    assert(NULL != m_pdOutData);  
    return m_pdOutData;  
}  
  
void PoolLayer::PrintOutputData()  
{  
    for (int i  = 0; i < m_iOutImageNumber; ++i)  
    {  
        cout << "pool image " << i  <<endl;  
        for (int m = 0; m < m_iOutImageEdge; ++m)  
        {  
            for (int n = 0; n < m_iOutImageEdge; ++n)  
            {  
                cout << m_pdOutData[i * m_iOutImageSize + m * m_iOutImageEdge + n] << ' ';  
            }  
            cout << endl;  
  
        }  
        cout <<endl;  
  
    }  
}  

