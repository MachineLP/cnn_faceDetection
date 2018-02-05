#ifndef NEURALNETWORK_H  
#define NEURALNETWORK_H  
  
#include "mlp.h"  
#include "cnn.h"  
#include <vector>  
using std::vector;  
  
/************************************************************************/  
/* ����һ�����������                                                                     */  
/************************************************************************/  
class NeuralNetWork  
{  
public:  
    NeuralNetWork(int iInput, int iOut);  
	NeuralNetWork();
    ~NeuralNetWork();  
    void Predict(double** in_data, int n);  
  
  
    double CalErrorRate(const vector<double *> &vecvalid, const vector<WORD> &vecValidlabel);  
  
    void Setwb(vector< vector<double*> > &vvAllw, vector< vector<double> > &vvAllb);  
    void SetTrainNum(int iNum);  
  
    int Predict(double *pInputData);  
    //    void Forward_propagation(double** ppdata, int n);  
    double* Forward_propagation(double *);  
  
private:  
    int m_iSampleNum; //��������  
    int m_iInput; //����ά��  
    int m_iOut; //���ά��  
      
    vector<CnnLayer *> vecCnns;   
    Mlp *m_pMlp;  
	

};   

int TestCnnTheano(const int iInputWidth, const int iOut, double *data, double &duration);
int TestCnnTheano24x24(const int iInputWidth, const int iOut, double *data, double &duration);
int TestCnnTheano20x20(const int iInputWidth, const int iOut, double *data, double &duration);
int TestCnnTheano12x12(const int iInputWidth, const int iOut, double *data, double &duration);
  
#endif 