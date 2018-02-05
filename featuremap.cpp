#include "stdafx.h"
#include "featuremap.h"  
#include "util.h"  
#include <cassert>  
  
FeatureMap::FeatureMap(int iInputImageNumber, int iInputImageWidth, int iFeatureMapNumber, int iKernelWidth):  
    m_iInputImageNumber(iInputImageNumber),  
    m_iInputImageWidth(iInputImageWidth),  
    m_iFeatureMapNumber(iFeatureMapNumber),  
    m_iKernelWidth(iKernelWidth)  
{  
    m_iFeatureMapWidth = m_iInputImageWidth - m_iKernelWidth + 1;  
    m_iInputImageSize = m_iInputImageWidth * m_iInputImageWidth;  
    m_iFeatureMapSize = m_iFeatureMapWidth * m_iFeatureMapWidth;  
      
  
    int iKernelSize;  
    iKernelSize = m_iKernelWidth * m_iKernelWidth;  
  
    double dbase = 1.0 / m_iInputImageSize;  
    srand((unsigned)time(NULL));  
      
  
    m_ppdWeigh = new double*[m_iFeatureMapNumber];  
    m_pdBias = new double[m_iFeatureMapNumber];  
  
    for (int i = 0; i < m_iFeatureMapNumber; ++i)  
    {  
        m_ppdWeigh[i] = new double[m_iInputImageNumber * iKernelSize];  
        for (int j = 0; j < m_iInputImageNumber * iKernelSize; ++j)  
        {  
            m_ppdWeigh[i][j] = uniform(-dbase, dbase);  
        }  
        //m_pdBias[i] = uniform(-dbase, dbase);  
        //theano�ľ����ò��û���õ�bias������pooling��ʹ��  
        m_pdBias[i] = 0;  
    }  
  
    m_pdOutputValue = new double[m_iFeatureMapNumber * m_iFeatureMapSize];  
      
//    m_dBias = uniform(-dbase, dbase);  
}  
  
FeatureMap::~FeatureMap()  
{  
    delete []m_pdOutputValue;  
    delete []m_pdBias;  
    for (int i = 0; i < m_iFeatureMapNumber; ++i)  
    {  
        delete []m_ppdWeigh[i];  
    }  
    delete []m_ppdWeigh;  
      
}  
  
void FeatureMap::SetWeigh(const vector<double *> &vecWeigh)  
{  
    assert(vecWeigh.size() == (DWORD)m_iFeatureMapNumber);  
    for (int i = 0; i < m_iFeatureMapNumber; ++i)  
    {  
        delete []m_ppdWeigh[i];  
        m_ppdWeigh[i] = vecWeigh[i];  
    }  
}  
  
/* 
 
������� 
pdInputData:һά�������������ɸ�����ͼ�� 
 
*/  
void FeatureMap::Convolute(double *pdInputData)  
{  
    for (int iMapIndex = 0; iMapIndex < m_iFeatureMapNumber; ++iMapIndex)  
    {  
        double dBias = m_pdBias[iMapIndex];  
        //ÿһ��featuremap  
        for (int i = 0; i < m_iFeatureMapWidth; ++i)  
        {  
            for (int j = 0; j < m_iFeatureMapWidth; ++j)  
            {  
                double dSum = 0.0;  
                int iInputIndex, iKernelIndex, iInputIndexStart, iKernelStart, iOutIndex;  
                //�����������������  
                iOutIndex = iMapIndex * m_iFeatureMapSize + i * m_iFeatureMapWidth + j;  
                //�ֱ����ÿһ������ͼ��  
                for (int k = 0; k < m_iInputImageNumber; ++k)  
                {  
                    //��kernel��Ӧ������ͼ�����ʼλ��  
                    //iInputIndexStart = k * m_iInputImageSize + j * m_iInputImageWidth + i;  
                    iInputIndexStart = k * m_iInputImageSize + i * m_iInputImageWidth + j;  
                    //kernel����ʼλ��  
                    iKernelStart = k * m_iKernelWidth * m_iKernelWidth;  
                    for (int m = 0; m < m_iKernelWidth; ++m)  
                    {  
                        for (int n = 0; n < m_iKernelWidth; ++n)  
                        {  
  
                            //iKernelIndex = iKernelStart + n * m_iKernelWidth + m;  
                            iKernelIndex = iKernelStart + m * m_iKernelWidth + n;  
                            //i am not sure, is the expression of below correct?  
                            iInputIndex = iInputIndexStart + m * m_iInputImageWidth + n;  
  
                            dSum += pdInputData[iInputIndex] * m_ppdWeigh[iMapIndex][iKernelIndex];  
                        }//end n  
                    }//end m  
  
                }//end k  
                //����ƫ��  
                //dSum += dBias;  
                m_pdOutputValue[iOutIndex] = dSum;        
                  
            }//end j  
        }//end i  
    }//end iMapIndex  
}  
  
void FeatureMap::PrintOutputData()  
{  
     for (int i  = 0; i < m_iFeatureMapNumber; ++i)  
     {  
         cout << "featuremap " << i <<endl;  
         for (int m = 0; m < m_iFeatureMapWidth; ++m)  
         {  
             for (int n = 0; n < m_iFeatureMapWidth; ++n)  
             {  
                 cout << m_pdOutputValue[i * m_iFeatureMapSize +m * m_iFeatureMapWidth +n] << ' ';  
             }  
             cout << endl;  
               
         }  
         cout <<endl;  
           
     }  
       
}  
