#ifndef NEURALBASE_H  
#define NEURALBASE_H  
  
#include <vector>  
using std::vector;  
typedef unsigned short WORD;  
  
class NeuralBase   
{  
public:  
    NeuralBase(int , int , int);  
    virtual ~NeuralBase();  
  
    virtual double* Forward_propagation(double* );  
    virtual void Back_propagation(double* , double* , double );  
  
    virtual void Train(double *x, WORD y, double dLr);  
    virtual int Predict(double *);  
  
  
    void Callbackwb();  
    void MakeOneLabel(int iMax, double *pdLabel);  
  
    void TrainAllSample(const vector<double*> &vecTrain, const vector<WORD> &vectrainlabel, double dLr);  
  
    double CalErrorRate(const vector<double*> &vecvalid, const vector<WORD> &vecValidlabel);  
  
    void Printwb();  
    void Writewb(const char *szName);  
    long Readwb(const char *szName, long);  
    void Setwb(vector<double*> &vpdw, vector<double> &vdb);  
    void PrintOutputData();  
  
  
    int m_iInput;  
    int m_iOut;  
    int m_iSamplenum;  
    double** m_ppdW;  
    double* m_pdBias;  
    //����ǰ�򴫲������ֵ��Ҳ�����յ�Ԥ��ֵ  
    double* m_pdOutdata;  
    //���򴫲�ʱ����ֵ  
    double* m_pdDelta;  
private:  
    void _callbackwb();  
  
};  
  
#endif // NEURALBASE_H  
