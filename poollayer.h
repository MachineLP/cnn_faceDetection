#ifndef POOLLAYER_H  
#define POOLLAYER_H  
#include <vector>  
using std::vector;   
class PoolLayer  
{  
public:  
    PoolLayer(int iOutImageNumber, int iPoolWidth, int iFeatureMapWidth);  
    ~PoolLayer();  
    void Convolute(double *pdInputData);  
    void SetBias(const vector<double> &vecBias);  
    double* GetOutputData();  
    void PrintOutputData();  
      
private:  
    int m_iOutImageNumber;  
    int m_iPoolWidth;  
    int m_iFeatureMapWidth;  
    int m_iPoolSize;  
    int m_iOutImageEdge;  
    int m_iOutImageSize;  
      
    double *m_pdOutData;  
    double *m_pdBias;  
};  
  
#endif // POOLLAYER_H  