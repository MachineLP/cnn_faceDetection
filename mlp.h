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
    int m_iSampleNum; //��������  
    int m_iInput; //����ά��  
    int m_iOut; //���ά��  
    int m_iHiddenLayerNum; //������Ŀ  
    int* m_piHiddenLayerSize; //�м�����Ĵ�С e.g. {3,4}��ʾ���������㣬��һ���������ڵ㣬�ڶ�����4���ڵ�  
  
      
};  
  
void mlp();  
void TestMlpTheano(const int m_iInput, const int ihidden, const int m_iOut);  
void TestMlpMnist(const int m_iInput, const int ihidden, const int m_iOut);  
  
#endif  
