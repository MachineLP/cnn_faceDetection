
#ifndef LOGISTICREGRESSIONLAYER  
#define LOGISTICREGRESSIONLAYER  
#include "neuralbase.h"  
  
typedef unsigned short WORD;  
  
class LogisticRegression: public NeuralBase  
{   
public:  
    LogisticRegression(int n_i, int i_o, int);  
    ~LogisticRegression();  
  
    double* Forward_propagation(double* input_data);  
    void Softmax(double* x);  
    void Train(double *pdTrain, WORD usLabel, double dLr);  
  
  
    void SetOldWb(double ppdWeigh[][3], double arriBias[8]);  
    int Predict(double *);  
  
    void MakeLabels(int* piMax, double (*pplabels)[8]);  
};  
  
void Test_lr();  
void Testwb();  
void Test_theano(const int m_iInput, const int m_iOut);  
#endif  
