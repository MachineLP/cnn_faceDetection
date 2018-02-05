#ifndef CNN_H  
#define CNN_H  
#include "featuremap.h"  
#include "poollayer.h"  
  
#include <vector>  
  
using std::vector;   
  
typedef unsigned short WORD;  
/** 
*本卷积模拟theano的测试过程 
*当输入层是num个featuremap时，本层卷积层假设有featureNum个featuremap。 
*对于本层每个像素点选取，上一层num个featuremap一起组合，并且没有bias 
*然后本层输出到pooling层，pooling只对poolsize内的像素取最大值，然后加上bias，总共有featuremap个bias值 
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
    //反向传播时所需值  
    double **m_ppdDelta;  
    double *m_pdInputData;  
    double *m_pdOutputData;  
};  
void TestCnn();  
  
#endif // CNN_H  
