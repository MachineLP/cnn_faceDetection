#include "stdafx.h"
#include <cmath>  
#include <cassert>  
#include <iomanip>  
#include <ctime>  
#include <iostream>  
#include "logisticRegression.h"   
#include "util.h"  
  
using namespace std;  
  
  
LogisticRegression::LogisticRegression(int n_i, int n_o, int n_t): NeuralBase(n_i, n_o, n_t)  
{  
  
}  
  
LogisticRegression::~LogisticRegression()  
{  
  
}  
  
  
  
void LogisticRegression::Softmax(double* x)  
{  
    double _max = 0.0;  
    double _sum = 0.0;  
  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        if(_max < x[i])  
            _max = x[i];  
    }  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        x[i] = exp(x[i]-_max);  
        _sum += x[i];  
    }  
  
    for(int i = 0; i < m_iOut; ++i)  
    {  
        x[i] /= _sum;  
    }  
}  
  
double* LogisticRegression::Forward_propagation(double* pdinputdata)  
{  
    NeuralBase::Forward_propagation(pdinputdata);  
    /************************************************************************/  
    /* 调试                                                                     */  
    /************************************************************************/  
    //cout << "Forward_propagation from   LogisticRegression" << endl;  
    //PrintOutputData();  
    //cout << "over\n";  
    Softmax(m_pdOutdata);  
    return m_pdOutdata;  
}  
  
  
int LogisticRegression::Predict(double *pdtest)  
{  
    Forward_propagation(pdtest);  
    /************************************************************************/  
    /* 调试使用                                                                     */  
    /************************************************************************/  
    //PrintOutputData();  
    int iResult = getMaxIndex(m_pdOutdata, m_iOut);  
  
    return iResult;  
  
}  
void LogisticRegression::Train(double *pdTrain, WORD usLabel, double dLr)  
{  
    Forward_propagation(pdTrain);  
    double *pdLabel = new double[m_iOut];  
    MakeOneLabel(usLabel, pdLabel);  
    Back_propagation(pdTrain, pdLabel, dLr);  
  
    delete []pdLabel;  
}  
//double LogisticRegression::CalErrorRate(const vector<double*> &vecvalid, const vector<WORD> &vecValidlabel)  
//{  
//    int iErrorNumber = 0, iValidNumber = vecValidlabel.size();  
//    for (int i = 0; i < iValidNumber; ++i)  
//    {  
//        int iResult = Predict(vecvalid[i]);  
//        if (iResult != vecValidlabel[i])  
//        {  
//            ++iErrorNumber;  
//        }  
//    }  
  
//    cout << "the num of error is " << iErrorNumber << endl;  
//    double dErrorRate = (double)iErrorNumber / iValidNumber;  
//    cout << "the error rate of Train sample by softmax is " << setprecision(10) << dErrorRate * 100 << "%" << endl;  
  
//    return dErrorRate;  
//}  
  
  
  
void LogisticRegression::SetOldWb(double ppdWeigh[][3], double arriBias[8])  
{  
    for (int i = 0; i < m_iOut; ++i)  
    {  
        for (int j = 0; j < m_iInput; ++j)  
            m_ppdW[i][j] = ppdWeigh[i][j];  
        m_pdBias[i] = arriBias[i];  
    }  
    cout << "Setwb----------" << endl;  
    printArrDouble(m_ppdW, m_iOut, m_iInput);  
    printArr(m_pdBias, m_iOut);  
}  
  
//void LogisticRegression::TrainAllSample(const vector<double*> &vecTrain, const vector<WORD> &vectrainlabel, double dLr)  
//{  
//    for (int j = 0; j < m_iSamplenum; ++j)  
//    {  
//        Train(vecTrain[j], vectrainlabel[j], dLr);  
//    }  
//}  
  
void LogisticRegression::MakeLabels(int* piMax, double (*pplabels)[8])  
{  
    for (int i = 0; i < m_iSamplenum; ++i)  
    {  
        for (int j = 0; j < m_iOut; ++j)  
            pplabels[i][j] = 0;  
        int k = piMax[i];  
        pplabels[i][k] = 1.0;  
    }  
}  
void Test_theano(const int m_iInput, const int m_iOut)  
{  
  
    // construct LogisticRegression  
    LogisticRegression classifier(m_iInput, m_iOut, 0);  
  
    vector<double*> vecTrain, vecvalid, vecw;  
    vector<double> vecb;  
    vector<WORD> vecValidlabel, vectrainlabel;  
  
    LoadTestSampleFromJson(vecvalid, vecValidlabel, "../.../../data/mnist.json", m_iInput);  
    LoadTestSampleFromJson(vecTrain, vectrainlabel, "../.../../data/mnisttrain.json", m_iInput);  
  
     // test  
  
    int itrainnum = vecTrain.size();  
    classifier.m_iSamplenum = itrainnum;  
  
    const int iepochs = 5;  
    const double dLr = 0.1;  
    for (int i = 0; i < iepochs; ++i)  
    {  
        classifier.TrainAllSample(vecTrain, vectrainlabel, dLr);  
  
        if (i % 2 == 0)  
        {  
            cout << "Predict------------" << i + 1 << endl;  
            classifier.CalErrorRate(vecvalid, vecValidlabel);  
  
        }  
    }  
  
  
  
    for (vector<double*>::iterator cit = vecTrain.begin(); cit != vecTrain.end(); ++cit)  
    {  
        delete [](*cit);  
    }  
    for (vector<double*>::iterator cit = vecvalid.begin(); cit != vecvalid.end(); ++cit)  
    {  
        delete [](*cit);  
    }  
}  
  
void Test_lr()  
{  
    srand(0);  
  
    double learning_rate = 0.1;  
    double n_epochs = 200;  
  
    int test_N = 2;  
    const int trainNum = 8, m_iInput = 3, m_iOut = 8;  
    //int m_iOut = 2;  
    double train_X[trainNum][m_iInput] = {  
        {1, 1, 1},  
        {1, 1, 0},  
        {1, 0, 1},  
        {1, 0, 0},  
        {0, 1, 1},  
        {0, 1, 0},  
        {0, 0, 1},  
        {0, 0, 0}  
    };  
    //sziMax存储的是最大值的下标  
    int sziMax[trainNum];  
    for (int i = 0; i < trainNum; ++i)  
        sziMax[i] = trainNum - i - 1;  
      
    // construct LogisticRegression  
    LogisticRegression classifier(m_iInput, m_iOut, trainNum);  
  
    // Train online  
    for(int epoch=0; epoch<n_epochs; epoch++) {  
        for(int i=0; i<trainNum; i++) {  
            //classifier.trainEfficient(train_X[i], train_Y[i], learning_rate);  
            classifier.Train(train_X[i], sziMax[i], learning_rate);  
        }  
    }  
  
    const char *pcfile = "test.wb";  
    classifier.Writewb(pcfile);  
  
    LogisticRegression logistic(m_iInput, m_iOut, trainNum);  
    logistic.Readwb(pcfile, 0);  
    // test data  
    double test_X[2][m_iOut] = {  
        {1, 0, 1},  
        {0, 0, 1}  
    };  
     // test  
    cout << "before Readwb ---------" << endl;  
    for(int i=0; i<test_N; i++) {  
        classifier.Predict(test_X[i]);  
        cout << endl;  
    }  
    cout << "after Readwb ---------" << endl;  
    for(int i=0; i<trainNum; i++) {  
        logistic.Predict(train_X[i]);  
        cout << endl;  
    }  
    cout << "*********\n";  
     
}  
void Testwb()  
{  
  
  
//    int test_N = 2;  
    const int trainNum = 8, m_iInput = 3, m_iOut = 8;  
    //int m_iOut = 2;  
    double train_X[trainNum][m_iInput] = {  
        {1, 1, 1},  
        {1, 1, 0},  
        {1, 0, 1},  
        {1, 0, 0},  
        {0, 1, 1},  
        {0, 1, 0},  
        {0, 0, 1},  
        {0, 0, 0}  
    };  
    double arriBias[m_iOut] = {1, 2, 3, 3, 3, 3, 2, 1};  
      
      
    // construct LogisticRegression  
    LogisticRegression classifier(m_iInput, m_iOut, trainNum);  
  
    classifier.SetOldWb(train_X, arriBias);  
  
    const char *pcfile = "test.wb";  
    classifier.Writewb(pcfile);  
  
    LogisticRegression logistic(m_iInput, m_iOut, trainNum);  
    logistic.Readwb(pcfile, 0);  
  
}  
