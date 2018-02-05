#include "stdafx.h"
#include <iostream>  
#include "mlp.h"  
#include "util.h"  
#include <cassert>  
#include <iomanip>  
  
using namespace std;  
  
const int m_iSamplenum = 8, innode = 3, outnode = 8;  
Mlp::Mlp(int n, int n_i, int n_o, int nhl, int *hls)  
{  
    m_iSampleNum = n;  
    m_iInput = n_i;  
    m_iOut = n_o;  
  
    m_iHiddenLayerNum = nhl;  
    m_piHiddenLayerSize = hls;  
  
    //构造网络结构  
    m_ppHiddenLayer = new HiddenLayer* [m_iHiddenLayerNum];  
    for(int i = 0; i < m_iHiddenLayerNum; ++i)  
    {  
        if(i == 0)  
        {  
            m_ppHiddenLayer[i] = new HiddenLayer(m_iInput, m_piHiddenLayerSize[i]);//第一个隐层  
        }  
        else  
        {  
            m_ppHiddenLayer[i] = new HiddenLayer(m_piHiddenLayerSize[i-1], m_piHiddenLayerSize[i]);//其他隐层  
        }  
    }  
    if (m_iHiddenLayerNum > 0)  
    {  
        m_pLogisticLayer = new LogisticRegression(m_piHiddenLayerSize[m_iHiddenLayerNum - 1], m_iOut, m_iSampleNum);//最后的softmax层  
    }  
    else  
    {  
        m_pLogisticLayer = new LogisticRegression(m_iInput, m_iOut, m_iSampleNum);//最后的softmax层  
  
    }  
      
}  
  
Mlp::~Mlp()  
{  
    //二维指针分配的对象不一定是二维数组  
    for(int i = 0; i < m_iHiddenLayerNum; ++i)  
        delete m_ppHiddenLayer[i];  //删除的时候不能加[]  
    delete[] m_ppHiddenLayer;  
    //log_layer只是一个普通的对象指针，不能作为数组delete  
    delete m_pLogisticLayer;//删除的时候不能加[]  
}  
void Mlp::TrainAllSample(const vector<double *> &vecTrain, const vector<WORD> &vectrainlabel, double dLr)  
{  
    cout << "Mlp::TrainAllSample" << endl;  
    for (int j = 0; j < m_iSampleNum; ++j)  
    {  
        Train(vecTrain[j], vectrainlabel[j], dLr);  
    }  
}  
  
void Mlp::Train(double *pdTrain, WORD usLabel, double dLr)  
{  
  
  
    //    cout << "******pdLabel****" << endl;  
    //  printArrDouble(ppdinLabel, m_iSampleNum, m_iOut);  
  
    double *pdLabel = new double[m_iOut];  
    MakeOneLabel(usLabel, pdLabel, m_iOut);  
  
    //前向传播阶段  
    for(int n = 0; n < m_iHiddenLayerNum; ++ n)  
    {  
        if(n == 0) //第一个隐层直接输入数据  
        {  
            m_ppHiddenLayer[n]->Forward_propagation(pdTrain);  
        }  
        else //其他隐层用前一层的输出作为输入数据  
        {  
            m_ppHiddenLayer[n]->Forward_propagation(m_ppHiddenLayer[n-1]->m_pdOutdata);  
        }  
    }  
    //softmax层使用最后一个隐层的输出作为输入数据  
    m_pLogisticLayer->Forward_propagation(m_ppHiddenLayer[m_iHiddenLayerNum-1]->m_pdOutdata);  
  
  
    //反向传播阶段  
    m_pLogisticLayer->Back_propagation(m_ppHiddenLayer[m_iHiddenLayerNum-1]->m_pdOutdata, pdLabel, dLr);  
  
    for(int n = m_iHiddenLayerNum-1; n >= 1; --n)  
    {  
        if(n == m_iHiddenLayerNum-1)  
        {  
            m_ppHiddenLayer[n]->Back_propagation(m_ppHiddenLayer[n-1]->m_pdOutdata,  
                    m_pLogisticLayer->m_pdDelta, m_pLogisticLayer->m_ppdW, m_pLogisticLayer->m_iOut, dLr);  
        }  
        else  
        {  
            double *pdInputData;  
            pdInputData = m_ppHiddenLayer[n-1]->m_pdOutdata;  
  
            m_ppHiddenLayer[n]->Back_propagation(pdInputData,  
                                                m_ppHiddenLayer[n+1]->m_pdDelta, m_ppHiddenLayer[n+1]->m_ppdW, m_ppHiddenLayer[n+1]->m_iOut, dLr);  
        }  
    }  
    //这里该怎么写？  
    if (m_iHiddenLayerNum > 1)  
        m_ppHiddenLayer[0]->Back_propagation(pdTrain,  
                                            m_ppHiddenLayer[1]->m_pdDelta, m_ppHiddenLayer[1]->m_ppdW, m_ppHiddenLayer[1]->m_iOut, dLr);  
    else  
        m_ppHiddenLayer[0]->Back_propagation(pdTrain,  
                                            m_pLogisticLayer->m_pdDelta, m_pLogisticLayer->m_ppdW, m_pLogisticLayer->m_iOut, dLr);  
  
  
    delete []pdLabel;  
}  
void Mlp::SetTrainNum(int iNum)  
{  
    m_iSampleNum = iNum;  
}  
  
double* Mlp::Forward_propagation(double* pData)  
{  
    double *pdForwardValue = pData;  
    for(int n = 0; n < m_iHiddenLayerNum; ++ n)  
    {  
        if(n == 0) //第一个隐层直接输入数据  
        {  
            pdForwardValue = m_ppHiddenLayer[n]->Forward_propagation(pData);  
        }  
        else //其他隐层用前一层的输出作为输入数据  
        {  
            pdForwardValue = m_ppHiddenLayer[n]->Forward_propagation(pdForwardValue);  
        }  
    }  
    return pdForwardValue;  
    //softmax层使用最后一个隐层的输出作为输入数据  
    //    m_pLogisticLayer->Forward_propagation(m_ppHiddenLayer[m_iHiddenLayerNum-1]->m_pdOutdata);  
  
    //    m_pLogisticLayer->Predict(m_ppHiddenLayer[m_iHiddenLayerNum-1]->m_pdOutdata);  
  
     
}  
  
int Mlp::Predict(double *pInputData)  
{  
    Forward_propagation(pInputData);  
  
    int iResult = m_pLogisticLayer->Predict(m_ppHiddenLayer[m_iHiddenLayerNum-1]->m_pdOutdata);  
  
    return iResult;  
  
}  
  
void Mlp::Setwb(vector< vector<double*> > &vvAllw, vector< vector<double> > &vvAllb)  
{  
    for (int i = 0; i < m_iHiddenLayerNum; ++i)  
    {  
        m_ppHiddenLayer[i]->Setwb(vvAllw[i], vvAllb[i]);  
    }  
    m_pLogisticLayer->Setwb(vvAllw[m_iHiddenLayerNum], vvAllb[m_iHiddenLayerNum]);  
}  
  
void Mlp::Writewb(const char *szName)  
{  
    for(int i = 0; i < m_iHiddenLayerNum; ++i)  
    {  
        m_ppHiddenLayer[i]->Writewb(szName);  
    }  
    m_pLogisticLayer->Writewb(szName);  
  
}  
double Mlp::CalErrorRate(const vector<double *> &vecvalid, const vector<WORD> &vecValidlabel)  
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
  
void Mlp::Readwb(const char *szName)  
{  
    long dcurpos = 0, dreadsize = 0;  
    for(int i = 0; i < m_iHiddenLayerNum; ++i)  
    {  
        dreadsize = m_ppHiddenLayer[i]->Readwb(szName, dcurpos);  
        cout << "hiddenlayer " << i + 1 << " read bytes: " << dreadsize << endl;  
        if (-1 != dreadsize)  
            dcurpos += dreadsize;  
        else  
        {  
            cout << "read wb error from HiddenLayer" << endl;  
            return;  
        }  
    }  
    dreadsize = m_pLogisticLayer->Readwb(szName, dcurpos);  
    if (-1 != dreadsize)  
        dcurpos += dreadsize;  
    else  
    {  
        cout << "read wb error from sofmaxLayer" << endl;  
        return;  
    }  
}  
  
int* Mlp::GetHiddenSize()  
{  
    return m_piHiddenLayerSize;  
}  
  
double* Mlp::GetHiddenOutputData()  
{  
    assert(m_iHiddenLayerNum > 0);  
    return m_ppHiddenLayer[m_iHiddenLayerNum-1]->m_pdOutdata;  
}  
  
int Mlp::GetHiddenNumber()  
{  
    return m_iHiddenLayerNum;  
}  
  
  
  
//double **makeLabelSample(double **label_x)  
double **makeLabelSample(double label_x[][outnode])  
{  
    double **pplabelSample;  
    pplabelSample = new double*[m_iSamplenum];  
    for (int i = 0; i < m_iSamplenum; ++i)  
    {  
        pplabelSample[i] = new double[outnode];  
    }  
  
    for (int i = 0; i < m_iSamplenum; ++i)  
    {  
        for (int j = 0; j < outnode; ++j)  
            pplabelSample[i][j] = label_x[i][j];  
    }  
    return pplabelSample;  
}  
double **maken_train(double train_x[][innode])  
{  
    double **ppn_train;  
    ppn_train = new double*[m_iSamplenum];  
    for (int i = 0; i < m_iSamplenum; ++i)  
    {  
        ppn_train[i] = new double[innode];  
    }  
  
    for (int i = 0; i < m_iSamplenum; ++i)  
    {  
        for (int j = 0; j < innode; ++j)  
            ppn_train[i][j] = train_x[i][j];  
    }  
    return ppn_train;  
}  
  
  
void TestMlpMnist(const int m_iInput, const int ihidden, const int m_iOut)  
{  
    const int ihiddenSize = 1;  
    int phidden[ihiddenSize] = {ihidden};  
    // construct LogisticRegression  
    Mlp neural(m_iSamplenum, m_iInput, m_iOut, ihiddenSize, phidden);  
  
  
    vector<double*> vecTrain, vecvalid;  
  
    vector<WORD> vecValidlabel, vectrainlabel;  
  
    LoadTestSampleFromJson(vecvalid, vecValidlabel, "../../data/mnist.json", m_iInput);  
    LoadTestSampleFromJson(vecTrain, vectrainlabel, "../../data/mnisttrain.json", m_iInput);  
  
    // test  
  
    int itrainnum = vecTrain.size();  
    neural.SetTrainNum(itrainnum);  
  
    const int iepochs = 1;  
    const double dLr = 0.1;  
    neural.CalErrorRate(vecvalid, vecValidlabel);  
  
    for (int i = 0; i < iepochs; ++i)  
    {  
        neural.TrainAllSample(vecTrain, vectrainlabel, dLr);  
        neural.CalErrorRate(vecvalid, vecValidlabel);  
  
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
  
  
void TestMlpTheano(const int m_iInput, const int ihidden, const int m_iOut)  
{  
  
    const int ihiddenSize = 1;  
    int phidden[ihiddenSize] = {ihidden};  
    // construct LogisticRegression  
    Mlp neural(m_iSamplenum, m_iInput, m_iOut, ihiddenSize, phidden);  
    vector<double*> vecTrain, vecw;  
    vector<double> vecb;  
    vector<WORD> vecLabel;  
    vector< vector<double*> > vvAllw;  
    vector< vector<double> > vvAllb;  
    const char *pcfilename = "../../data/theanomlp.json";  
      
    vector<int> vecSecondDimOfWeigh;  
    vecSecondDimOfWeigh.push_back(m_iInput);  
    vecSecondDimOfWeigh.push_back(ihidden);  
  
    LoadWeighFromJson(vvAllw, vvAllb, pcfilename, vecSecondDimOfWeigh);  
  
    LoadTestSampleFromJson(vecTrain, vecLabel, "../../data/mnist_validall.json", m_iInput);  
  
  
    cout << "loadwb ---------" << endl;  
  
    int itrainnum = vecTrain.size();  
    neural.SetTrainNum(itrainnum);  
  
  
    neural.Setwb(vvAllw, vvAllb);  
  
    cout << "Predict------------" << endl;  
    neural.CalErrorRate(vecTrain, vecLabel);  
      
  
    for (vector<double*>::iterator cit = vecTrain.begin(); cit != vecTrain.end(); ++cit)  
    {  
        delete [](*cit);  
    }  
  
}  
  
void mlp()  
{  
    //输入样本  
    double X[m_iSamplenum][innode]= {  
        {0,0,0},{0,0,1},{0,1,0},{0,1,1},{1,0,0},{1,0,1},{1,1,0},{1,1,1}  
    };  
  
    double Y[m_iSamplenum][outnode]={  
        {1, 0, 0, 0, 0, 0, 0, 0},  
        {0, 1, 0, 0, 0, 0, 0, 0},  
        {0, 0, 1, 0, 0, 0, 0, 0},  
        {0, 0, 0, 1, 0, 0, 0, 0},  
        {0, 0, 0, 0, 1, 0, 0, 0},  
        {0, 0, 0, 0, 0, 1, 0, 0},  
        {0, 0, 0, 0, 0, 0, 1, 0},  
        {0, 0, 0, 0, 0, 0, 0, 1},  
    };  
    WORD pdLabel[outnode] = {0, 1, 2, 3, 4, 5, 6, 7};  
    const int ihiddenSize = 2;  
    int phidden[ihiddenSize] = {5, 5};  
    //printArr(phidden, 1);  
    Mlp neural(m_iSamplenum, innode, outnode, ihiddenSize, phidden);  
    double **train_x, **ppdLabel;  
    train_x = maken_train(X);  
    //printArrDouble(train_x, m_iSamplenum, innode);  
    ppdLabel = makeLabelSample(Y);  
    for (int i = 0; i < 3500; ++i)  
    {  
        for (int j = 0; j < m_iSamplenum; ++j)  
        {  
            neural.Train(train_x[j], pdLabel[j], 0.1);  
        }  
    }  
  
    cout<<"trainning complete..."<<endl;  
    for (int i = 0; i < m_iSamplenum; ++i)  
        neural.Predict(train_x[i]);  
    //szName存放权值  
    //  const char *szName = "mlp55new.wb";  
    //  neural.Writewb(szName);  
  
  
    //  Mlp neural2(m_iSamplenum, innode, outnode, ihiddenSize, phidden);  
    //  cout<<"Readwb start..."<<endl;  
    //  neural2.Readwb(szName);  
    //  cout<<"Readwb end..."<<endl;  
  
    //    cout << "----------after readwb________" << endl;  
    //    for (int i = 0; i < m_iSamplenum; ++i)  
    //        neural2.Forward_propagation(train_x[i]);  
  
    for (int i = 0; i != m_iSamplenum; ++i)  
    {  
        delete []train_x[i];  
        delete []ppdLabel[i];  
    }  
    delete []train_x;  
    delete []ppdLabel;  
    cout<<endl;  
}  
