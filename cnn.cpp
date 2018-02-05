#include "stdafx.h"
#include "cnn.h"  
#include "util.h"  
#include <cassert>  
  
CnnLayer::CnnLayer(int iSampleNum, int iInputImageNumber, int iInputImageWidth, int iFeatureMapNumber,  
                   int iKernelWidth, int iPoolWidth):  
    m_iSampleNum(iSampleNum), m_pdInputData(NULL), m_pdOutputData(NULL)  
{  
    m_pFeatureMap = new FeatureMap(iInputImageNumber, iInputImageWidth, iFeatureMapNumber, iKernelWidth);  
    int iFeatureMapWidth =  iInputImageWidth - iKernelWidth + 1;  
    m_pPoolLayer = new PoolLayer(iFeatureMapNumber, iPoolWidth, iFeatureMapWidth);  
     
}  
  
CnnLayer::~CnnLayer()  
{  
    delete m_pFeatureMap;  
    delete m_pPoolLayer;  
}  
  
void CnnLayer::Forward_propagation(double *pdInputData)  
{  
    m_pFeatureMap->Convolute(pdInputData);  
    m_pPoolLayer->Convolute(m_pFeatureMap->GetFeatureMapValue());  
    m_pdOutputData = m_pPoolLayer->GetOutputData();  
    /************************************************************************/  
    /* 调试卷积过程的各阶段结果，调通后删除                                                                     */  
    /************************************************************************/  
    /*m_pFeatureMap->PrintOutputData(); 
    m_pPoolLayer->PrintOutputData();*/  
}  
  
void CnnLayer::SetInputAllData(double **ppInputAllData, int iInputNum)  
{  
      
}  
  
double* CnnLayer::GetOutputData()  
{  
    assert(NULL != m_pdOutputData);  
  
    return m_pdOutputData;  
}  
  
void CnnLayer::Setwb(vector<double*> &vpdw, vector<double> &vdb)  
{  
    m_pFeatureMap->SetWeigh(vpdw);  
    m_pPoolLayer->SetBias(vdb);  
      
}  
  
void CnnLayer::SetTrainNum( int iSampleNumber )  
{  
    m_iSampleNum = iSampleNumber;  
}  
  
void CnnLayer::PrintOutputData()  
{  
    m_pFeatureMap->PrintOutputData();  
    m_pPoolLayer->PrintOutputData();  
  
}  
void TestCnn()  
{  
    const int iFeatureMapNumber = 2, iPoolWidth = 2, iInputImageWidth = 8, iKernelWidth = 3, iInputImageNumber = 2;  
  
    double *pdImage = new double[iInputImageWidth * iInputImageWidth * iInputImageNumber];  
    double arrInput[iInputImageNumber][iInputImageWidth * iInputImageWidth];  
  
    MakeCnnSample(arrInput, pdImage, iInputImageWidth, iInputImageNumber);  
  
    double *pdKernel = new double[3 * 3 * iInputImageNumber];  
    double arrKernel[3 * 3 * iInputImageNumber];  
    MakeCnnWeigh(pdKernel, iInputImageNumber) ;  
      
  
    CnnLayer cnn(3, iInputImageNumber, iInputImageWidth, iFeatureMapNumber, iKernelWidth, iPoolWidth);  
  
    vector <double*> vecWeigh;  
    vector <double> vecBias;  
    for (int i = 0; i < iFeatureMapNumber; ++i)  
    {  
        vecBias.push_back(1.0);  
    }  
    vecWeigh.push_back(pdKernel);  
    for (int i = 0; i < 3 * 3 * 2; ++i)  
    {  
        arrKernel[i] = i;  
    }  
    vecWeigh.push_back(arrKernel);  
    cnn.Setwb(vecWeigh, vecBias);  
      
    cnn.Forward_propagation(pdImage);  
    cnn.PrintOutputData();  
      
  
  
    delete []pdKernel;  
    delete []pdImage;    
}  
