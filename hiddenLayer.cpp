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
/* ��Ҫע����ǣ� 
���Ϊ�˸���theano�Ĳ��Խ������ô���ز�ļ����Ҫѡ��tanh�� 
����Ϊ��mlp��ѵ�����̣������Ҫѡ��sigmoid                                                                     */  
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
    pdInputData          Ϊ�������� 
    *pdNextLayerDelta   Ϊ��һ��Ĳв�ֵdelta,��һ����СΪiNextLayerOutNum������ 
    **ppdnextLayerW      Ϊ�˲㵽��һ���Ȩֵ 
    iNextLayerOutNum    ʵ���Ͼ�����һ���n_out 
    dLr                  Ϊѧϰ��learning rate 
    m_iSampleNum                   Ϊѵ���������� 
    */  
  
    //sigmaԪ�ظ���Ӧ�뱾�㵥Ԫ����һ�£������ϴ�������  
    //������û���Լ����԰������԰�  
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
    //����õ�����Ĳв�delta  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        m_pdDelta[i] = sigma[i] * m_pdOutdata[i] * (1 - m_pdOutdata[i]);  
    }  
  
    //���������Ȩֵw  
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
