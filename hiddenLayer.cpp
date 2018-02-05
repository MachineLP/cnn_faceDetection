#include "stdafx.h"
#include <cmath>  
#include <cassert>  
#include <cstdlib>  
#include <ctime>  
#include <iostream>  
#include "hiddenLayer.h"  
#include "util.h"  
  
using namespace std;  
  
  
  
HiddenLayer::HiddenLayer(int n_i, int n_o): NeuralBase(n_i, n_o, 0)  
{  
}  
  
HiddenLayer::~HiddenLayer()  
{  
  
}  
  
/************************************************************************/  
/* 需要注意的是： 
如果为了复现theano的测试结果，那么隐藏层的激活函数要选用tanh； 
否则，为了mlp的训练过程，激活函数要选择sigmoid                                                                     */  
/************************************************************************/  
double* HiddenLayer::Forward_propagation(double* pdInputData)  
{  
    NeuralBase::Forward_propagation(pdInputData);  
    for(int i = 0; i < m_iOut; ++i)  
    {  
       // m_pdOutdata[i] = sigmoid(m_pdOutdata[i]);  
        m_pdOutdata[i] = mytanh(m_pdOutdata[i]);  
  
    }  
    return m_pdOutdata;  
}  
  
void HiddenLayer::Back_propagation(double *pdInputData, double *pdNextLayerDelta,  
                                   double** ppdnextLayerW, int iNextLayerOutNum, double dLr)  
{  
    /* 
    pdInputData          为输入数据 
    *pdNextLayerDelta   为下一层的残差值delta,是一个大小为iNextLayerOutNum的数组 
    **ppdnextLayerW      为此层到下一层的权值 
    iNextLayerOutNum    实际上就是下一层的n_out 
    dLr                  为学习率learning rate 
    m_iSampleNum                   为训练样本总数 
    */  
  
    //sigma元素个数应与本层单元个数一致，而网上代码有误  
    //作者是没有自己测试啊，测试啊  
    //double* sigma = new double[iNextLayerOutNum];  
    double* sigma = new double[m_iOut];  
    //double sigma[10];  
    for(int i = 0; i < m_iOut; ++i)  
        sigma[i] = 0.0;  
  
    for(int i = 0; i < iNextLayerOutNum; ++i)  
    {  
        for(int j = 0; j < m_iOut; ++j)  
        {  
            sigma[j] += ppdnextLayerW[i][j] * pdNextLayerDelta[i];  
        }  
    }  
    //计算得到本层的残差delta  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        m_pdDelta[i] = sigma[i] * m_pdOutdata[i] * (1 - m_pdOutdata[i]);  
    }  
  
    //调整本层的权值w  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        for(int j = 0; j < m_iInput; ++j)  
        {  
            m_ppdW[i][j] += dLr * m_pdDelta[i] * pdInputData[j];  
        }  
        m_pdBias[i] += dLr * m_pdDelta[i];  
    }  
    delete[] sigma;  
}  
