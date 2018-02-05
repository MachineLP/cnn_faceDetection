
#include "stdafx.h"
#include "neuralNetwork.h"  
  
#include <iostream>  
#include "util.h"  
#include <iomanip>  
#include <time.h>
  
using namespace std;  

  
NeuralNetWork::NeuralNetWork(int iInput, int iOut):m_iSampleNum(0), m_iInput(iInput), m_iOut(iOut), m_pMlp(NULL)  
{  
    int iFeatureMapNumber = 20, iPoolWidth = 4, iInputImageWidth = m_iInput, iKernelWidth = 5, iInputImageNumber = 1;
  
	// 只用一层，并且全连接层也去掉了，直接连Logistic Recession。
    CnnLayer *pCnnLayer = new CnnLayer(m_iSampleNum, iInputImageNumber, iInputImageWidth, iFeatureMapNumber, iKernelWidth, iPoolWidth);  
    vecCnns.push_back(pCnnLayer);  
  
    //iInputImageNumber = 2;  
    //iInputImageWidth = 18;  
    //iFeatureMapNumber = 5;  
    //pCnnLayer = new CnnLayer(m_iSampleNum, iInputImageNumber, iInputImageWidth, iFeatureMapNumber, iKernelWidth, iPoolWidth);  
    //vecCnns.push_back(pCnnLayer);  
  
	iInputImageWidth = (m_iInput - 5 + 1) / 4;
    const int ihiddenSize = 1;  
    int phidden[ihiddenSize] = {500};
    // construct LogisticRegression  
    m_pMlp = new Mlp(m_iSampleNum, iFeatureMapNumber * iInputImageWidth * iInputImageWidth, m_iOut, /*0*/ ihiddenSize, phidden);
  
}  
  
NeuralNetWork::~NeuralNetWork()  
{  
  
    for (vector<CnnLayer*>::iterator it = vecCnns.begin(); it != vecCnns.end(); ++it)  
    {  
        delete *it;  
    }  
    delete m_pMlp;  
}  
  
void NeuralNetWork::SetTrainNum(int iNum)  
{  
    m_iSampleNum = iNum;  
  
    for (size_t i = 0; i < vecCnns.size(); ++i)  
    {  
        vecCnns[i]->SetTrainNum(iNum);  
    }  
    m_pMlp->SetTrainNum(iNum);  
      
}  
  
int NeuralNetWork::Predict(double *pdInputdata)  
{  
    double *pdPredictData = NULL;  
    pdPredictData = Forward_propagation(pdInputdata);  
  
    int iResult = -1;  
      
      
    iResult = m_pMlp->m_pLogisticLayer->Predict(pdPredictData);  
  
    return iResult;  
}  
  
double* NeuralNetWork::Forward_propagation(double *pdInputData)  
{  
    double *pdPredictData = pdInputData;  
  
    vector<CnnLayer*>::iterator it;  
    CnnLayer *pCnnLayer;  
    for (it = vecCnns.begin(); it != vecCnns.end(); ++it)  
    {  
        pCnnLayer = *it;  
        pCnnLayer->Forward_propagation(pdPredictData);  
        pdPredictData = pCnnLayer->GetOutputData();  
    }  
    //此时pCnnLayer指向最后一个卷积层,pdInputData是卷积层的最后输出  
    //暂时忽略mlp的前向计算，以后加上  
    // pdPredictData = m_pMlp->Forward_propagation(pdPredictData);
    pdPredictData = m_pMlp->Forward_propagation(pdPredictData);
    return pdPredictData;  
}  
  
void NeuralNetWork::Setwb(vector< vector<double*> > &vvAllw, vector< vector<double> > &vvAllb)  
{  
    for (size_t i = 0; i < vecCnns.size(); ++i)  
    {  
        vecCnns[i]->Setwb(vvAllw[i], vvAllb[i]);  
    }  
      
    size_t iLayerNum = vvAllw.size();  
    //for (size_t i = vecCnns.size(); i < iLayerNum - 1; ++i)  
    //{  
    //    int iHiddenIndex = 0;  
    //    m_pMlp->m_ppHiddenLayer[iHiddenIndex]->Setwb(vvAllw[i], vvAllb[i]);  
    //    ++iHiddenIndex;  
    //}
    for (size_t i = vecCnns.size(); i < iLayerNum - 1; ++i)
    {
        int iHiddenIndex = 0;
        m_pMlp->m_ppHiddenLayer[iHiddenIndex]->Setwb(vvAllw[i], vvAllb[i]);
        ++iHiddenIndex;
    }
    m_pMlp->m_pLogisticLayer->Setwb(vvAllw[iLayerNum - 1], vvAllb[iLayerNum - 1]);
}  
double NeuralNetWork::CalErrorRate(const vector<double *> &vecvalid, const vector<WORD> &vecValidlabel)  
{  
    cout << "Predict------------" << endl;  
    int iErrorNumber = 0, iValidNumber = vecValidlabel.size();  
    //iValidNumber = 1;  
    for (int i = 0; i < iValidNumber; ++i)  
    {  
        int iResult = Predict(vecvalid[i]);  
        //cout << i << ",valid is " << iResult << " label is " << vecValidlabel[i] << endl;  
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
  
/************************************************************************/  
/*  
测试样本采用mnist库，此cnn的结构与theano教程上的一致，即 
输入是28*28图像，接下来是2个卷积层（卷积+pooling），featuremap个数分别是20和50， 
然后是全连接层（500个神经元），最后输出层10个神经元 
 
*/  
/************************************************************************/  
 
int TestCnnTheano(const int iInputWidth, const int iOut, double *data, double &duration)  
{  
	//构建卷积神经网络  
    static NeuralNetWork neural(iInputWidth, iOut);  
	cout << "loadwb ---------" << endl;  

	//存取theano的权值  
	static vector< vector<double*> > vvAllw;  
	static vector< vector<double> > vvAllb;  
	//保存theano权值与测试样本的文件  
	const char *szTheanoWeigh = "/Users/liupeng/Desktop/my/faceDetection-cnn/faceDetection-cnn/data/zInput40/theanocnn.json";  
	//将每次权值的第二维（列宽）保存到vector中，用于读取json文件  
	vector<int> vecSecondDimOfWeigh;  
	vecSecondDimOfWeigh.push_back(5 * 5);    
	vecSecondDimOfWeigh.push_back(20 * 9 * 9);
	vecSecondDimOfWeigh.push_back(500);

	static int i = 0;
	i++;
	if(i == 1)
	{
		LoadWeighFromJson(vvAllw, vvAllb, szTheanoWeigh, vecSecondDimOfWeigh);
		//将权值设置到cnn中  
		neural.Setwb(vvAllw, vvAllb); 
	}
	double t1, t2;
     
    t1 = 1;//GetTickCount();
    //前向计算cnn的错误率，输出结果  
	int ret = neural.Predict(data);
    t2 = 2;//GetTickCount();
	duration = ((double)(t2 - t1));

	return ret;
} 

int TestCnnTheano24x24(const int iInputWidth, const int iOut, double *data, double &duration)
{
	//构建卷积神经网络  
    static NeuralNetWork neural(iInputWidth, iOut);  
	cout << "loadwb ---------" << endl;  

	//存取theano的权值  
	static vector< vector<double*> > vvAllw;  
	static vector< vector<double> > vvAllb;  
	//保存theano权值与测试样本的文件  
	const char *szTheanoWeigh = "/Users/liupeng/Desktop/my/faceDetection-cnn/faceDetection-cnn/data/zInput24/theanocnn.json";
	//将每次权值的第二维（列宽）保存到vector中，用于读取json文件  
	vector<int> vecSecondDimOfWeigh;  
	vecSecondDimOfWeigh.push_back(5 * 5);    
	vecSecondDimOfWeigh.push_back(2 * 5 * 5); 
	// vecSecondDimOfWeigh.push_back(100); 

	static int i = 0;
	i++;
	if(i == 1)
	{
		LoadWeighFromJson(vvAllw, vvAllb, szTheanoWeigh, vecSecondDimOfWeigh);
		//将权值设置到cnn中  
		neural.Setwb(vvAllw, vvAllb); 
	}
	double t1, t2;
     
    t1 = 1;//GetTickCount();
    //前向计算cnn的错误率，输出结果  
	int ret = neural.Predict(data);
    t2 = 2;//GetTickCount();
	duration = ((double)(t2 - t1));

	return ret;
}

int TestCnnTheano20x20(const int iInputWidth, const int iOut, double *data, double &duration)
{
	//构建卷积神经网络  
    static NeuralNetWork neural(iInputWidth, iOut);  
	cout << "loadwb ---------" << endl;  

	//存取theano的权值  
	static vector< vector<double*> > vvAllw;  
	static vector< vector<double> > vvAllb;  
	//保存theano权值与测试样本的文件  
	const char *szTheanoWeigh = "/Users/liupeng/Desktop/my/faceDetection-cnn/faceDetection-cnn/data/zInput20/theanocnn.json";
	//将每次权值的第二维（列宽）保存到vector中，用于读取json文件  
	vector<int> vecSecondDimOfWeigh;  
	vecSecondDimOfWeigh.push_back(5 * 5);    
	vecSecondDimOfWeigh.push_back(2 * 4 * 4); 
	// vecSecondDimOfWeigh.push_back(100); 

	static int i = 0;
	i++;
	if(i == 1)
	{
		LoadWeighFromJson(vvAllw, vvAllb, szTheanoWeigh, vecSecondDimOfWeigh);
		//将权值设置到cnn中  
		neural.Setwb(vvAllw, vvAllb); 
	}
	double t1, t2;
     
    t1 = 1;//GetTickCount();
    //前向计算cnn的错误率，输出结果  
	int ret = neural.Predict(data);
    t2 = 2;//GetTickCount();
	duration = ((double)(t2 - t1));

	return ret;
}

int TestCnnTheano12x12(const int iInputWidth, const int iOut, double *data, double &duration)
{
	//构建卷积神经网络  
    static NeuralNetWork neural(iInputWidth, iOut);  
	cout << "loadwb ---------" << endl;  

	//存取theano的权值  
	static vector< vector<double*> > vvAllw;  
	static vector< vector<double> > vvAllb;  
	//保存theano权值与测试样本的文件  
	const char *szTheanoWeigh = "/Users/liupeng/Desktop/my/faceDetection-cnn/faceDetection-cnn/data/zInput12/theanocnn.json";
	//将每次权值的第二维（列宽）保存到vector中，用于读取json文件  
	vector<int> vecSecondDimOfWeigh;  
	vecSecondDimOfWeigh.push_back(5 * 5);    
	vecSecondDimOfWeigh.push_back(2 * 2 * 2); 
	// vecSecondDimOfWeigh.push_back(100); 

	static int i = 0;
	i++;
	if(i == 1)
	{
		LoadWeighFromJson(vvAllw, vvAllb, szTheanoWeigh, vecSecondDimOfWeigh);
		//将权值设置到cnn中  
		neural.Setwb(vvAllw, vvAllb); 
	}
	double t1, t2;
     
    t1 = 1;//GetTickCount();
    //前向计算cnn的错误率，输出结果  
	int ret = neural.Predict(data);
    t2 = 2;//GetTickCount();
	duration = ((double)(t2 - t1));

	return ret;
}
