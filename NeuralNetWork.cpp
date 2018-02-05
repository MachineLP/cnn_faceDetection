
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
  
	// ֻ��һ�㣬����ȫ���Ӳ�Ҳȥ���ˣ�ֱ����Logistic Recession��
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
    //��ʱpCnnLayerָ�����һ�������,pdInputData�Ǿ�����������  
    //��ʱ����mlp��ǰ����㣬�Ժ����  
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
������������mnist�⣬��cnn�Ľṹ��theano�̳��ϵ�һ�£��� 
������28*28ͼ�񣬽�������2������㣨���+pooling����featuremap�����ֱ���20��50�� 
Ȼ����ȫ���Ӳ㣨500����Ԫ������������10����Ԫ 
 
*/  
/************************************************************************/  
 
int TestCnnTheano(const int iInputWidth, const int iOut, double *data, double &duration)  
{  
	//�������������  
    static NeuralNetWork neural(iInputWidth, iOut);  
	cout << "loadwb ---------" << endl;  

	//��ȡtheano��Ȩֵ  
	static vector< vector<double*> > vvAllw;  
	static vector< vector<double> > vvAllb;  
	//����theanoȨֵ������������ļ�  
	const char *szTheanoWeigh = "/Users/liupeng/Desktop/my/faceDetection-cnn/faceDetection-cnn/data/zInput40/theanocnn.json";  
	//��ÿ��Ȩֵ�ĵڶ�ά���п����浽vector�У����ڶ�ȡjson�ļ�  
	vector<int> vecSecondDimOfWeigh;  
	vecSecondDimOfWeigh.push_back(5 * 5);    
	vecSecondDimOfWeigh.push_back(20 * 9 * 9);
	vecSecondDimOfWeigh.push_back(500);

	static int i = 0;
	i++;
	if(i == 1)
	{
		LoadWeighFromJson(vvAllw, vvAllb, szTheanoWeigh, vecSecondDimOfWeigh);
		//��Ȩֵ���õ�cnn��  
		neural.Setwb(vvAllw, vvAllb); 
	}
	double t1, t2;
     
    t1 = 1;//GetTickCount();
    //ǰ�����cnn�Ĵ����ʣ�������  
	int ret = neural.Predict(data);
    t2 = 2;//GetTickCount();
	duration = ((double)(t2 - t1));

	return ret;
} 

int TestCnnTheano24x24(const int iInputWidth, const int iOut, double *data, double &duration)
{
	//�������������  
    static NeuralNetWork neural(iInputWidth, iOut);  
	cout << "loadwb ---------" << endl;  

	//��ȡtheano��Ȩֵ  
	static vector< vector<double*> > vvAllw;  
	static vector< vector<double> > vvAllb;  
	//����theanoȨֵ������������ļ�  
	const char *szTheanoWeigh = "/Users/liupeng/Desktop/my/faceDetection-cnn/faceDetection-cnn/data/zInput24/theanocnn.json";
	//��ÿ��Ȩֵ�ĵڶ�ά���п����浽vector�У����ڶ�ȡjson�ļ�  
	vector<int> vecSecondDimOfWeigh;  
	vecSecondDimOfWeigh.push_back(5 * 5);    
	vecSecondDimOfWeigh.push_back(2 * 5 * 5); 
	// vecSecondDimOfWeigh.push_back(100); 

	static int i = 0;
	i++;
	if(i == 1)
	{
		LoadWeighFromJson(vvAllw, vvAllb, szTheanoWeigh, vecSecondDimOfWeigh);
		//��Ȩֵ���õ�cnn��  
		neural.Setwb(vvAllw, vvAllb); 
	}
	double t1, t2;
     
    t1 = 1;//GetTickCount();
    //ǰ�����cnn�Ĵ����ʣ�������  
	int ret = neural.Predict(data);
    t2 = 2;//GetTickCount();
	duration = ((double)(t2 - t1));

	return ret;
}

int TestCnnTheano20x20(const int iInputWidth, const int iOut, double *data, double &duration)
{
	//�������������  
    static NeuralNetWork neural(iInputWidth, iOut);  
	cout << "loadwb ---------" << endl;  

	//��ȡtheano��Ȩֵ  
	static vector< vector<double*> > vvAllw;  
	static vector< vector<double> > vvAllb;  
	//����theanoȨֵ������������ļ�  
	const char *szTheanoWeigh = "/Users/liupeng/Desktop/my/faceDetection-cnn/faceDetection-cnn/data/zInput20/theanocnn.json";
	//��ÿ��Ȩֵ�ĵڶ�ά���п����浽vector�У����ڶ�ȡjson�ļ�  
	vector<int> vecSecondDimOfWeigh;  
	vecSecondDimOfWeigh.push_back(5 * 5);    
	vecSecondDimOfWeigh.push_back(2 * 4 * 4); 
	// vecSecondDimOfWeigh.push_back(100); 

	static int i = 0;
	i++;
	if(i == 1)
	{
		LoadWeighFromJson(vvAllw, vvAllb, szTheanoWeigh, vecSecondDimOfWeigh);
		//��Ȩֵ���õ�cnn��  
		neural.Setwb(vvAllw, vvAllb); 
	}
	double t1, t2;
     
    t1 = 1;//GetTickCount();
    //ǰ�����cnn�Ĵ����ʣ�������  
	int ret = neural.Predict(data);
    t2 = 2;//GetTickCount();
	duration = ((double)(t2 - t1));

	return ret;
}

int TestCnnTheano12x12(const int iInputWidth, const int iOut, double *data, double &duration)
{
	//�������������  
    static NeuralNetWork neural(iInputWidth, iOut);  
	cout << "loadwb ---------" << endl;  

	//��ȡtheano��Ȩֵ  
	static vector< vector<double*> > vvAllw;  
	static vector< vector<double> > vvAllb;  
	//����theanoȨֵ������������ļ�  
	const char *szTheanoWeigh = "/Users/liupeng/Desktop/my/faceDetection-cnn/faceDetection-cnn/data/zInput12/theanocnn.json";
	//��ÿ��Ȩֵ�ĵڶ�ά���п����浽vector�У����ڶ�ȡjson�ļ�  
	vector<int> vecSecondDimOfWeigh;  
	vecSecondDimOfWeigh.push_back(5 * 5);    
	vecSecondDimOfWeigh.push_back(2 * 2 * 2); 
	// vecSecondDimOfWeigh.push_back(100); 

	static int i = 0;
	i++;
	if(i == 1)
	{
		LoadWeighFromJson(vvAllw, vvAllb, szTheanoWeigh, vecSecondDimOfWeigh);
		//��Ȩֵ���õ�cnn��  
		neural.Setwb(vvAllw, vvAllb); 
	}
	double t1, t2;
     
    t1 = 1;//GetTickCount();
    //ǰ�����cnn�Ĵ����ʣ�������  
	int ret = neural.Predict(data);
    t2 = 2;//GetTickCount();
	duration = ((double)(t2 - t1));

	return ret;
}
