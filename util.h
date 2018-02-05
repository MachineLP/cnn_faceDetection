#ifndef UTIL_H  
#define UTIL_H  
#include <iostream>  
#include <cstdio>  
#include <cstdlib>  
#include <ctime>  
#include <vector>   
  
using namespace std;  
  
  
typedef unsigned char BYTE;  
typedef unsigned short WORD;
typedef unsigned int DWORD;
  
double sigmoid(double x);  
double mytanh(double dx);  
  
typedef struct stShapeWb  
{  
    stShapeWb(int w, int h):width(w), height(h){}  
    int width;  
    int height;  
}ShapeWb_S;  
  
void MakeOneLabel(int iMax, double *pdLabel, int m_iOut);  
  
double uniform(double _min, double _max);  
//void printArr(T *parr, int num);  
//void printArrDouble(double **pparr, int row, int col);  
void initArr(double *parr, int num);  
int getMaxIndex(double *pdarr, int num);  
void Printivec(const vector<int> &ivec);  
void savewb(const char *szName, double **m_ppdW, double *m_pdBias,  
             int irow, int icol);  
long loadwb(const char *szName, double **m_ppdW, double *m_pdBias,  
            int irow, int icol, long dstartpos);  
  
void TestLoadJson(const char *pcfilename);  
bool LoadvtFromJson(vector<double*> &vecTrain, vector<WORD> &vecLabel, const char *filename, const int m_iInput);  
bool LoadwbFromJson(vector<double*> &vecTrain, vector<double> &vecLabel, const char *filename, const int m_iInput);  
bool LoadTestSampleFromJson(vector<double*> &vecTrain, vector<WORD> &vecLabel, const char *filename, const int m_iInput);  
bool LoadwbByByte(vector<double*> &vecTrain, vector<double> &vecLabel, const char *filename, const int m_iInput);  
bool LoadallwbByByte(vector< vector<double*> > &vvAllw, vector< vector<double> > &vvAllb, const char *filename,  
                     const int m_iInput, const int ihidden, const int m_iOut);  
bool LoadWeighFromJson(vector< vector<double*> > &vvAllw, vector< vector<double> > &vvAllb,  
                     const char *filename, const vector<int> &vecSecondDimOfWeigh);  
void MakeCnnSample(double arr[2][64], double *pdImage, int iImageWidth, int iNumOfImage );  
void MakeCnnWeigh(double *, int iNumOfKernel);  
template <typename T>  
void printArr(T *parr, int num)  
{  
    cout << "****printArr****" << endl;  
  
    for (int i = 0; i < num; ++i)  
        cout << parr[i] << ' ';  
    cout << endl;  
}  
template <typename T>  
void printArrDouble(T **pparr, int row, int col)  
{  
    cout << "****printArrDouble****" << endl;  
    for (int i = 0; i < row; ++i)  
    {  
        for (int j = 0; j < col; ++j)  
        {  
            cout << pparr[i][j] << ' ';  
        }  
        cout << endl;  
    }  
}  
  
  
#endif  
