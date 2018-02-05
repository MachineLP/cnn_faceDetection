#ifndef MLP_H  
#define MLP_H  
  
#include "hiddenLayer.h"   
#include "logisticRegression.h"  
  
  
  
class Mlp  
{  
public:  
    Mlp(int n, int n_i, int n_o, int nhl, int *hls);  
    ~Mlp();  
  
    //    void Train(double** in_data, double** in_label, double dLr, int epochs);  
    void Predict(double** in_data, int n);  
    void Train(double *x, WORD y, double dLr);  
    void TrainAllSample(const vector<double*> &vecTrain, const vector<WORD> &vectrainlabel, double dLr);  
    double CalErrorRate(const vector<double *> &vecvalid, const vector<WORD> &vecValidlabel);  
    void Writewb(const char *szName);  
    void Readwb(const char *szName);  
  
    void Setwb(vector< vector<double*> > &vvAllw, vector< vector<double> > &vvAllb);  
    void SetTrainNum(int iNum);  
  
    int Predict(double *pInputData);  
    //    void Forward_propagation(double** ppdata, int n);  
    double* Forward_propagation(double *);  
  
    int* GetHiddenSize();  
    int GetHiddenNumber();  
    double *GetHiddenOutputData();  
  
    HiddenLayer **m_ppHiddenLayer;  
    LogisticRegression *m_pLogisticLayer;  
  
private:  
    int m_iSampleNum; //样本数量  
    int m_iInput; //输入维数  
    int m_iOut; //输出维数  
    int m_iHiddenLayerNum; //隐层数目  
    int* m_piHiddenLayerSize; //中间隐层的大小 e.g. {3,4}表示有两个隐层，第一个有三个节点，第二个有4个节点  
  
      
};  
  
void mlp();  
void TestMlpTheano(const int m_iInput, const int ihidden, const int m_iOut);  
void TestMlpMnist(const int m_iInput, const int ihidden, const int m_iOut);  
  
#endif  
