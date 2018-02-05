#include "stdafx.h"
#include "neuralbase.h"  
  
#include <cmath>  
#include <cassert>  
#include <ctime>  
   
#include <iomanip>  
#include <iostream>  
  
  
#include "util.h"  
  
using namespace std;  
  
  
NeuralBase::NeuralBase(int n_i, int n_o, int n_t):m_iInput(n_i), m_iOut(n_o), m_iSamplenum(n_t)  
{  
    m_ppdW = new double* [m_iOut];  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        m_ppdW[i] = new double [m_iInput];  
    }  
    m_pdBias = new double [m_iOut];  
  
    double a = 1.0 / m_iInput;  
  
    srand((unsigned)time(NULL));  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        for(int j = 0; j < m_iInput; ++j)  
            m_ppdW[i][j] = uniform(-a, a);  
        m_pdBias[i] = uniform(-a, a);  
    }  
  
    m_pdDelta = new double [m_iOut];  
    m_pdOutdata = new double [m_iOut];  
}  
  
NeuralBase::~NeuralBase()  
{  
    Callbackwb();  
    delete[] m_pdOutdata;  
    delete[] m_pdDelta;  
}  
void NeuralBase::Callbackwb()  
{  
    _callbackwb();  
}  
double NeuralBase::CalErrorRate(const vector<double *> &vecvalid, const vector<WORD> &vecValidlabel)  
{  
    int iErrorNumber = 0, iValidNumber = vecValidlabel.size();  
    for (int i = 0; i < iValidNumber; ++i)  
    {  
        int iResult = Predict(vecvalid[i]);  
        if (iResult != vecValidlabel[i])  
        {  
            ++iErrorNumber;  
        }  
    }  
  
    cout << "the num of error is " << iErrorNumber << endl;  
    double dErrorRate = (double)iErrorNumber / iValidNumber;  
    cout << "the error rate of Train sample by softmax is " << setprecision(10) << dErrorRate * 100 << "%" << endl;  
  
    return dErrorRate;  
}  
int NeuralBase::Predict(double *)  
{  
    cout << "NeuralBase::Predict(double *)" << endl;  
    return 0;  
}  
  
void NeuralBase::_callbackwb()  
{  
    for(int i=0; i < m_iOut; i++)  
        delete []m_ppdW[i];  
    delete[] m_ppdW;  
    delete[] m_pdBias;  
}  
  
void NeuralBase::Printwb()  
{  
    cout << "'****m_ppdW****\n";  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        for(int j = 0; j < m_iInput; ++j)  
            cout << m_ppdW[i][j] << ' ';  
        cout << endl;  
    }  
    cout << "'****m_pdBias****\n";  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        cout << m_pdBias[i] << ' ';  
    }  
    cout << endl;  
    cout << "'****output****\n";  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        cout << m_pdOutdata[i] << ' ';  
    }  
    cout << endl;  
  
}  
  
  
double* NeuralBase::Forward_propagation(double* input_data)  
{  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        m_pdOutdata[i] = 0.0;  
        for(int j = 0; j < m_iInput; ++j)  
        {  
            m_pdOutdata[i] += m_ppdW[i][j]*input_data[j];  
        }  
        m_pdOutdata[i] += m_pdBias[i];  
    }  
    return m_pdOutdata;  
}  
  
void NeuralBase::Back_propagation(double* input_data, double* pdlabel, double dLr)  
{  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        m_pdDelta[i] = pdlabel[i] - m_pdOutdata[i] ;  
        for(int j = 0; j < m_iInput; ++j)  
        {  
            m_ppdW[i][j] += dLr * m_pdDelta[i] * input_data[j] / m_iSamplenum;  
        }  
        m_pdBias[i] += dLr * m_pdDelta[i] / m_iSamplenum;  
    }  
}  
void NeuralBase::MakeOneLabel(int imax, double *pdlabel)  
{  
    for (int j = 0; j < m_iOut; ++j)  
        pdlabel[j] = 0;  
    pdlabel[imax] = 1.0;  
}  
void NeuralBase::Writewb(const char *szName)  
{  
    savewb(szName, m_ppdW, m_pdBias, m_iOut, m_iInput);  
}  
long NeuralBase::Readwb(const char *szName, long dstartpos)  
{  
    return loadwb(szName, m_ppdW, m_pdBias, m_iOut, m_iInput, dstartpos);  
}  
  
void NeuralBase::Setwb(vector<double*> &vpdw, vector<double> &vdb)  
{  
    assert(vpdw.size() == (DWORD)m_iOut);
    for (int i = 0; i < m_iOut; ++i)  
    {  
        delete []m_ppdW[i];  
        m_ppdW[i] = vpdw[i];  
        m_pdBias[i] = vdb[i];  
    }  
}  
  
void NeuralBase::TrainAllSample(const vector<double *> &vecTrain, const vector<WORD> &vectrainlabel, double dLr)  
{  
    for (int j = 0; j < m_iSamplenum; ++j)  
    {  
        Train(vecTrain[j], vectrainlabel[j], dLr);  
    }  
}  
void NeuralBase::Train(double *x, WORD y, double dLr)  
{  
    (void)x;  
    (void)y;  
    (void)dLr;  
    cout << "NeuralBase::Train(double *x, WORD y, double dLr)" << endl;  
}  
  
void NeuralBase::PrintOutputData()  
{  
    for (int i = 0; i < m_iOut; ++i)  
    {  
        cout << m_pdOutdata[i] << ' ';  
    }  
    cout << endl;  
      
  
}  
