#ifndef CNN_H  
#define CNN_H  
#include "featuremap.h"  
#include "poollayer.h"  
  
#include <vector>  
  
using std::vector;   
  
typedef unsigned short WORD;  
/** 
*�����ģ��theano�Ĳ��Թ��� 
*���������num��featuremapʱ���������������featureNum��featuremap�� 
*���ڱ���ÿ�����ص�ѡȡ����һ��num��featuremapһ����ϣ�����û��bias 
*Ȼ�󱾲������pooling�㣬poolingֻ��poolsize�ڵ�����ȡ���ֵ��Ȼ�����bias���ܹ���featuremap��biasֵ 
*/  
class CnnLayer  
{  
public:  
    CnnLayer(int iSampleNum, int iInputImageNumber, int iInputImageWidth, int iFeatureMapNumber,  
        int iKernelWidth, int iPoolWidth);  
    ~CnnLayer();  
    void Forward_propagation(double *pdInputData);  
    void Back_propagation(double* , double* , double );  
  
    void Train(double *x, WORD y, double dLr);  
    int Predict(double *);  
  
    void Setwb(vector<double*> &vpdw, vector<double> &vdb);  
    void SetInputAllData(double **ppInputAllData, int iInputNum);  
    void SetTrainNum(int iSampleNumber);  
    void PrintOutputData();  
  
    double* GetOutputData();  
private:  
    int m_iSampleNum;  
  
    FeatureMap *m_pFeatureMap;  
    PoolLayer *m_pPoolLayer;  
    //���򴫲�ʱ����ֵ  
    double **m_ppdDelta;  
    double *m_pdInputData;  
    double *m_pdOutputData;  
};  
void TestCnn();  
  
#endif // CNN_H  
