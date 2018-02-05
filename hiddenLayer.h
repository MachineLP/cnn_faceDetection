#ifndef HIDDENLAYER_H  
#define HIDDENLAYER_H  
   
#include "neuralbase.h"  
  
class HiddenLayer: public NeuralBase  
{  
public:  
    HiddenLayer(int n_i, int n_o);  
    ~HiddenLayer();  
  
    double* Forward_propagation(double* input_data);  
      
    void Back_propagation(double *pdInputData, double *pdNextLayerDelta,  
                          double** ppdnextLayerW, int iNextLayerOutNum, double dLr);  
      
  
  
};  
  
#endif  