#ifndef FEATUREMAP_H  
#define FEATUREMAP_H  
  
#include <cassert>  
#include <vector>  
using std::vector;  
  
class FeatureMap  
{  
public:  
    FeatureMap(int iInputImageNumber, int iInputImageWidth, int iFeatureMapNumber, int iKernelWidth);  
    ~FeatureMap();  
    void Forward_propagation(double* );  
    void Back_propagation(double* , double* , double );   
  
  
    void Convolute(double *pdInputData);  
  
    int GetFeatureMapSize()  
    {  
        return m_iFeatureMapSize;  
    }  
  
    int GetFeatureMapWidth()  
    {  
        return m_iFeatureMapWidth;  
    }  
  
    double* GetFeatureMapValue()  
    {  
        assert(m_pdOutputValue != NULL);  
        return m_pdOutputValue;  
    }  
  
    void SetWeigh(const vector<double *> &vecWeigh);  
    void PrintOutputData();  
  
    double **m_ppdWeigh;  
    double *m_pdBias;  
  
private:  
    int m_iInputImageNumber;  
    int m_iInputImageWidth;  
    int m_iInputImageSize;  
    int m_iFeatureMapNumber;  
    int m_iFeatureMapWidth;  
    int m_iFeatureMapSize;  
    int m_iKernelWidth;  
      
//    double m_dBias;  
    double *m_pdOutputValue;  
};  
  
#endif // FEATUREMAP_H  
