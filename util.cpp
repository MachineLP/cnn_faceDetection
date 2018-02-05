#include "stdafx.h"
#include "util.h"  
#include <iostream>  
#include <ctime>  
#include <cmath>   
#include <cassert>  
#include <fstream>  
#include <cstring>  
#include <stack>  
#include <iomanip>  
  
using namespace std;  
  
int getMaxIndex(double *pdarr, int num)  
{  
    double dmax = -1;  
    int iMax = -1;  
    for(int i = 0; i < num; ++i)  
    {  
        if (pdarr[i] > dmax)  
        {  
            dmax = pdarr[i];  
            iMax = i;  
        }  
    }  
    return iMax;  
}  
  
//double sigmoid(double dx)  
//{  
//    return 1.0/(1.0+exp(-dx));  
//}  
double mytanh(double dx)  
{  
    double e2x = exp(2 * dx);  
    return (e2x - 1) / (e2x + 1);  
}  
  
double uniform(double _min, double _max)  
{  
    return rand()/(RAND_MAX + 1.0) * (_max - _min) + _min;  
}  
  
void initArr(double *parr, int num)  
{  
    for (int i = 0; i < num; ++i)  
        parr[i] = 0.0;  
}  
  
  
  
void savewb(const char *szName, double **m_ppdW, double *m_pdBias,  
             int irow, int icol)  
{  
    FILE *pf;  
    if( (pf = fopen(szName, "ab" )) == NULL )   
    {   
        printf( "File coulkd not be opened " );   
        return;  
    }   
  
    int isizeofelem = sizeof(double);  
    for (int i = 0; i < irow; ++i)  
    {  
        if (fwrite((const void*)m_ppdW[i], isizeofelem, icol, pf) != icol)  
        {  
            fputs ("Writing m_ppdW error",stderr);  
            return;  
        }  
    }  
    if (fwrite((const void*)m_pdBias, isizeofelem, irow, pf) != irow)  
    {  
        fputs ("Writing m_ppdW error",stderr);  
        return;  
    }  
    fclose(pf);  
}  
long loadwb(const char *szName, double **m_ppdW, double *m_pdBias,  
            int irow, int icol, long dstartpos)  
{  
    FILE *pf;  
    long dtotalbyte = 0, dreadsize;  
    if( (pf = fopen(szName, "rb" )) == NULL )   
    {   
        printf( "File coulkd not be opened " );   
        return -1;  
    }   
    //让文件指针偏移到正确位置  
    fseek(pf, dstartpos , SEEK_SET);  
  
    int isizeofelem = sizeof(double);  
    for (int i = 0; i < irow; ++i)  
    {  
        dreadsize = fread((void*)m_ppdW[i], isizeofelem, icol, pf);  
        if (dreadsize != icol)  
        {  
            fputs ("Reading m_ppdW error",stderr);  
            return -1;  
        }  
        //每次成功读取，都要加到dtotalbyte中，最后返回  
        dtotalbyte += dreadsize;  
    }  
    dreadsize = fread(m_pdBias, isizeofelem, irow, pf);  
    if (dreadsize != irow)  
    {  
        fputs ("Reading m_pdBias error",stderr);  
        return -1;  
    }  
    dtotalbyte += dreadsize;  
    dtotalbyte *= isizeofelem;  
    fclose(pf);  
    return dtotalbyte;    
}  
  
void Printivec(const vector<int> &ivec)  
{  
    for (vector<int>::const_iterator it = ivec.begin(); it != ivec.end(); ++it)  
    {  
        cout << *it << ' ';  
    }  
    cout << endl;  
}  
void TestLoadJson(const char *pcfilename)  
{  
    vector<double *> vpdw;  
    vector<double> vdb;  
    vector< vector<double*> > vvAllw;  
    vector< vector<double> > vvAllb;  
    int m_iInput = 28 * 28, ihidden = 500, m_iOut = 10;  
    LoadallwbByByte(vvAllw, vvAllb, pcfilename, m_iInput, ihidden, m_iOut );  
  
  
}  
  
//read vt from mnist, format is [[[], [],..., []],[1, 3, 5,..., 7]]  
bool LoadvtFromJson(vector<double*> &vecTrain, vector<WORD> &vecLabel, const char *filename, const int m_iInput)  
{  
    cout << "loadvtFromJson" << endl;  
    const int ciStackSize = 10;  
    const int ciFeaturesize = m_iInput;  
    int arriStack[ciStackSize], iTop = -1;  
  
    ifstream ifs;  
    ifs.open(filename, ios::in);  
    assert(ifs.is_open());  
    BYTE ucRead, ucLeftbrace, ucRightbrace, ucComma, ucSpace;  
    ucLeftbrace = '[';  
    ucRightbrace = ']';  
    ucComma = ',';  
    ucSpace = '0';  
    ifs >> ucRead;  
    assert(ucRead == ucLeftbrace);  
    //栈中全部存放左括号，用1代表,0说明清除  
    arriStack[++iTop] = 1;  
    //样本train开始  
    ifs >> ucRead;  
    assert(ucRead == ucLeftbrace);  
    arriStack[++iTop] = 1;//iTop is 1  
    int iIndex;  
    bool isdigit = false;  
    double dread, *pdvt;  
    //load vt sample  
    while (iTop > 0)  
    {  
        if (isdigit == false)  
        {  
            ifs >> ucRead;  
            isdigit = true;  
            if (ucRead == ucComma)  
            {  
                //next char is space or leftbrace  
                //                ifs >> ucRead;  
                isdigit = false;  
                continue;  
            }  
            if (ucRead == ucSpace)  
            {  
                //if pdvt is null, next char is leftbrace;  
                //else next char is double value  
                if (pdvt == NULL)  
                    isdigit = false;  
                continue;  
            }  
            if (ucRead == ucLeftbrace)  
            {  
                pdvt = new double[ciFeaturesize];  
                memset(pdvt, 0, ciFeaturesize * sizeof(double));  
                //iIndex数组下标  
                iIndex = 0;  
                arriStack[++iTop] = 1;  
                continue;  
            }  
  
            if (ucRead == ucRightbrace)  
            {  
                if (pdvt != NULL)  
                {  
                    assert(iIndex == ciFeaturesize);  
                    vecTrain.push_back(pdvt);  
                    pdvt = NULL;  
                }  
                isdigit = false;  
                arriStack[iTop--] = 0;  
                continue;  
            }  
        }  
        else  
        {  
            ifs >> dread;  
            pdvt[iIndex++] = dread;  
            isdigit = false;  
        }  
    };  
    //next char is dot  
    ifs >> ucRead;  
    assert(ucRead == ucComma);  
    cout << vecTrain.size() << endl;  
    //读取label  
    WORD usread;  
    isdigit = false;  
    while (iTop > -1 && ifs.eof() == false)  
    {  
        if (isdigit == false)  
        {  
            ifs >> ucRead;  
            isdigit = true;  
            if (ucRead == ucComma)  
            {  
                //next char is space or leftbrace  
                //                ifs >> ucRead;  
//                isdigit = false;  
                continue;  
            }  
            if (ucRead == ucSpace)  
            {  
                //if pdvt is null, next char is leftbrace;  
                //else next char is double value  
                if (pdvt == NULL)  
                    isdigit = false;  
                continue;  
            }  
            if (ucRead == ucLeftbrace)  
            {  
                arriStack[++iTop] = 1;  
                continue;  
            }  
  
            //右括号的下一个字符是右括号（最后一个字符）  
            if (ucRead == ucRightbrace)  
            {  
                isdigit = false;  
                arriStack[iTop--] = 0;  
                continue;  
            }  
        }  
        else  
        {  
            ifs >> usread;  
            vecLabel.push_back(usread);  
            isdigit = false;  
        }  
    };  
    assert(vecLabel.size() == vecTrain.size());  
    assert(iTop == -1);  
  
    ifs.close();  
  
  
    return true;  
}  
bool testjsonfloat(const char *filename)  
{  
    vector<double> vecTrain;  
    cout << "testjsondouble" << endl;  
    const int ciStackSize = 10;  
    int arriStack[ciStackSize], iTop = -1;  
  
    ifstream ifs;  
    ifs.open(filename, ios::in);  
    assert(ifs.is_open());  
  
    BYTE ucRead, ucLeftbrace, ucRightbrace, ucComma;  
    ucLeftbrace = '[';  
    ucRightbrace = ']';  
    ucComma = ',';  
    ifs >> ucRead;  
    assert(ucRead == ucLeftbrace);  
    //栈中全部存放左括号，用1代表,0说明清除  
    arriStack[++iTop] = 1;  
    //样本train开始  
    ifs >> ucRead;  
    assert(ucRead == ucLeftbrace);  
    arriStack[++iTop] = 1;//iTop is 1  
    double fread;  
    bool isdigit = false;  
  
  
    while (iTop > -1)  
    {  
        if (isdigit == false)  
        {  
            ifs >> ucRead;  
            isdigit = true;  
            if (ucRead == ucComma)  
            {  
                //next char is space or leftbrace  
                //                ifs >> ucRead;  
                isdigit = false;  
                continue;  
            }  
            if (ucRead == ' ')  
                continue;  
            if (ucRead == ucLeftbrace)  
            {  
                arriStack[++iTop] = 1;  
                continue;  
            }  
  
            if (ucRead == ucRightbrace)  
            {  
                isdigit = false;  
                //右括号的下一个字符是右括号（最后一个字符）  
                arriStack[iTop--] = 0;  
                continue;  
            }  
        }  
        else  
        {  
            ifs >> fread;  
            vecTrain.push_back(fread);  
            isdigit = false;  
        }  
    }  
  
    ifs.close();  
  
  
    return true;  
  
}  
  
bool LoadwbFromJson(vector<double*> &vecTrain, vector<double> &vecLabel, const char *filename, const int m_iInput)  
{  
    cout << "loadvtFromJson" << endl;  
    const int ciStackSize = 10;  
    const int ciFeaturesize = m_iInput;  
    int arriStack[ciStackSize], iTop = -1;  
  
    ifstream ifs;  
    ifs.open(filename, ios::in);  
    assert(ifs.is_open());  
    BYTE ucRead, ucLeftbrace, ucRightbrace, ucComma, ucSpace;  
    ucLeftbrace = '[';  
    ucRightbrace = ']';  
    ucComma = ',';  
    ucSpace = '0';  
    ifs >> ucRead;  
    assert(ucRead == ucLeftbrace);  
    //栈中全部存放左括号，用1代表,0说明清除  
    arriStack[++iTop] = 1;  
    //样本train开始  
    ifs >> ucRead;  
    assert(ucRead == ucLeftbrace);  
    arriStack[++iTop] = 1;//iTop is 1  
    int iIndex;  
    bool isdigit = false;  
    double dread, *pdvt;  
    //load vt sample  
    while (iTop > 0)  
    {  
        if (isdigit == false)  
        {  
            ifs >> ucRead;  
            isdigit = true;  
            if (ucRead == ucComma)  
            {  
                //next char is space or leftbrace  
                //                ifs >> ucRead;  
                isdigit = false;  
                continue;  
            }  
            if (ucRead == ucSpace)  
            {  
                //if pdvt is null, next char is leftbrace;  
                //else next char is double value  
                if (pdvt == NULL)  
                    isdigit = false;  
                continue;  
            }  
            if (ucRead == ucLeftbrace)  
            {  
                pdvt = new double[ciFeaturesize];  
                memset(pdvt, 0, ciFeaturesize * sizeof(double));  
                //iIndex数组下标  
                iIndex = 0;  
                arriStack[++iTop] = 1;  
                continue;  
            }  
  
            if (ucRead == ucRightbrace)  
            {  
                if (pdvt != NULL)  
                {  
                    assert(iIndex == ciFeaturesize);  
                    vecTrain.push_back(pdvt);  
                    pdvt = NULL;  
                }  
                isdigit = false;  
                arriStack[iTop--] = 0;  
                continue;  
            }  
        }  
        else  
        {  
            ifs >> dread;  
            pdvt[iIndex++] = dread;  
            isdigit = false;  
        }  
    };  
    //next char is dot  
    ifs >> ucRead;  
    assert(ucRead == ucComma);  
    cout << vecTrain.size() << endl;  
    //读取label  
    double usread;  
    isdigit = false;  
    while (iTop > -1 && ifs.eof() == false)  
    {  
        if (isdigit == false)  
        {  
            ifs >> ucRead;  
            isdigit = true;  
            if (ucRead == ucComma)  
            {  
                //next char is space or leftbrace  
                //                ifs >> ucRead;  
//                isdigit = false;  
                continue;  
            }  
            if (ucRead == ucSpace)  
            {  
                //if pdvt is null, next char is leftbrace;  
                //else next char is double value  
                if (pdvt == NULL)  
                    isdigit = false;  
                continue;  
            }  
            if (ucRead == ucLeftbrace)  
            {  
                arriStack[++iTop] = 1;  
                continue;  
            }  
  
            //右括号的下一个字符是右括号（最后一个字符）  
            if (ucRead == ucRightbrace)  
            {  
                isdigit = false;  
                arriStack[iTop--] = 0;  
                continue;  
            }  
        }  
        else  
        {  
            ifs >> usread;  
            vecLabel.push_back(usread);  
            isdigit = false;  
        }  
    };  
    assert(vecLabel.size() == vecTrain.size());  
    assert(iTop == -1);  
  
    ifs.close();  
  
  
    return true;  
}  
bool vec2double(vector<BYTE> &vecDigit, double &dvalue)  
{  
    if (vecDigit.empty())  
        return false;  
    int ivecsize = vecDigit.size();  
    const int iMaxlen = 50;  
    char szdigit[iMaxlen];  
    assert(iMaxlen > ivecsize);  
    memset(szdigit, 0, iMaxlen);  
    int i;  
    for (i = 0; i < ivecsize; ++i)  
        szdigit[i] = vecDigit[i];  
    szdigit[i++] = '\0';  
    vecDigit.clear();  
    dvalue = atof(szdigit);  
  
    return true;  
}  
bool vec2short(vector<BYTE> &vecDigit, WORD &usvalue)  
{  
    if (vecDigit.empty())  
        return false;  
    int ivecsize = vecDigit.size();  
    const int iMaxlen = 50;  
    char szdigit[iMaxlen];  
    assert(iMaxlen > ivecsize);  
    memset(szdigit, 0, iMaxlen);  
    int i;  
    for (i = 0; i < ivecsize; ++i)  
        szdigit[i] = vecDigit[i];  
    szdigit[i++] = '\0';  
    vecDigit.clear();  
    usvalue = atoi(szdigit);  
    return true;  
}  
void readDigitFromJson(ifstream &ifs, vector<double*> &vecTrain, vector<WORD> &vecLabel,  
                       vector<BYTE> &vecDigit, double *&pdvt, int &iIndex,  
                       const int ciFeaturesize, int *arrStack, int &iTop, bool bFirstlist)  
{  
    BYTE ucRead;  
    WORD usvalue;  
    double dvalue;  
  
  
    const BYTE ucLeftbrace = '[', ucRightbrace = ']', ucComma = ',', ucSpace = ' ';  
  
    ifs.read((char*)(&ucRead), 1);  
    switch (ucRead)  
    {  
        case ucLeftbrace:  
        {  
            if (bFirstlist)  
            {  
  
                pdvt = new double[ciFeaturesize];  
                memset(pdvt, 0, ciFeaturesize * sizeof(double));  
                iIndex = 0;  
            }  
            arrStack[++iTop] = 1;  
            break;  
        }  
        case ucComma:  
        {  
            //next char is space or leftbrace  
            if (bFirstlist)  
            {  
                if (vecDigit.empty() == false)  
                {  
                    vec2double(vecDigit, dvalue);  
                    pdvt[iIndex++] = dvalue;  
                }  
            }  
            else  
            {  
                if(vec2short(vecDigit, usvalue))  
                    vecLabel.push_back(usvalue);  
            }  
            break;  
        }  
        case ucSpace:  
            break;  
        case ucRightbrace:  
        {  
            if (bFirstlist)  
            {  
                if (pdvt != NULL)  
                {  
  
                    vec2double(vecDigit, dvalue);  
                    pdvt[iIndex++] = dvalue;  
                    vecTrain.push_back(pdvt);  
                    pdvt = NULL;  
                }  
                assert(iIndex == ciFeaturesize);  
            }  
            else  
            {  
                if(vec2short(vecDigit, usvalue))  
                    vecLabel.push_back(usvalue);  
            }  
            arrStack[iTop--] = 0;  
            break;  
        }  
        default:  
        {  
            vecDigit.push_back(ucRead);  
            break;  
        }  
    }  
  
}  
void readDoubleFromJson(ifstream &ifs, vector<double*> &vecTrain, vector<double> &vecLabel,  
                       vector<BYTE> &vecDigit, double *&pdvt, int &iIndex,  
                       const int ciFeaturesize, int *arrStack, int &iTop, bool bFirstlist)  
{  
    BYTE ucRead;  
    double dvalue;  
  
  
    const BYTE ucLeftbrace = '[', ucRightbrace = ']', ucComma = ',', ucSpace = ' ';  
  
    ifs.read((char*)(&ucRead), 1);  
    switch (ucRead)  
    {  
        case ucLeftbrace:  
        {  
            if (bFirstlist)  
            {  
  
                pdvt = new double[ciFeaturesize];  
                memset(pdvt, 0, ciFeaturesize * sizeof(double));  
                iIndex = 0;  
            }  
            arrStack[++iTop] = 1;  
            break;  
        }  
        case ucComma:  
        {  
            //next char is space or leftbrace  
            if (bFirstlist)  
            {  
                if (vecDigit.empty() == false)  
                {  
                    vec2double(vecDigit, dvalue);  
                    pdvt[iIndex++] = dvalue;  
                }  
            }  
            else  
            {  
                if(vec2double(vecDigit, dvalue))  
                    vecLabel.push_back(dvalue);  
            }  
            break;  
        }  
        case ucSpace:  
            break;  
        case ucRightbrace:  
        {  
            if (bFirstlist)  
            {  
                if (pdvt != NULL)  
                {  
  
                    vec2double(vecDigit, dvalue);  
                    pdvt[iIndex++] = dvalue;  
                    vecTrain.push_back(pdvt);  
                    pdvt = NULL;  
                }  
                assert(iIndex == ciFeaturesize);  
            }  
            else  
            {  
                if(vec2double(vecDigit, dvalue))  
                    vecLabel.push_back(dvalue);  
            }  
            arrStack[iTop--] = 0;  
            break;  
        }  
        default:  
        {  
            vecDigit.push_back(ucRead);  
            break;  
        }  
    }  
  
}  
  
bool LoadallwbByByte(vector< vector<double*> > &vvAllw, vector< vector<double> > &vvAllb, const char *filename,  
                     const int m_iInput, const int ihidden, const int m_iOut)  
{  
    cout << "LoadallwbByByte" << endl;  
    const int szistsize = 10;  
    int ciFeaturesize = m_iInput;  
    const BYTE ucLeftbrace = '[', ucRightbrace = ']', ucComma = ',', ucSpace = ' ';  
    int arrStack[szistsize], iTop = -1, iIndex = 0;  
  
  
    ifstream ifs;  
    ifs.open(filename, ios::in | ios::binary);  
    assert(ifs.is_open());  
  
  
    double *pdvt;  
    BYTE ucRead;  
    ifs.read((char*)(&ucRead), 1);  
    assert(ucRead == ucLeftbrace);  
    //栈中全部存放左括号，用1代表,0说明清除  
    arrStack[++iTop] = 1;  
    ifs.read((char*)(&ucRead), 1);  
    assert(ucRead == ucLeftbrace);  
    arrStack[++iTop] = 1;//iTop is 1  
    ifs.read((char*)(&ucRead), 1);  
    assert(ucRead == ucLeftbrace);  
    arrStack[++iTop] = 1;//iTop is 2  
    vector<BYTE> vecDigit;  
    vector<double *> vpdw;  
    vector<double> vdb;  
    while (iTop > 1 && ifs.eof() == false)  
    {  
        readDoubleFromJson(ifs, vpdw, vdb, vecDigit, pdvt, iIndex, m_iInput, arrStack, iTop, true);  
    };  
    //next char is dot  
    ifs.read((char*)(&ucRead), 1);;  
    assert(ucRead == ucComma);  
    cout << vpdw.size() << endl;  
    //next char is space  
    while (iTop > 0 && ifs.eof() == false)  
    {  
        readDoubleFromJson(ifs, vpdw, vdb, vecDigit, pdvt, iIndex, m_iInput, arrStack, iTop, false);  
    };  
    assert(vpdw.size() == vdb.size());  
    assert(iTop == 0);  
    vvAllw.push_back(vpdw);  
    vvAllb.push_back(vdb);  
    //clear vpdw and pdb 's contents  
    vpdw.clear();  
    vdb.clear();  
  
    //next char is comma  
    ifs.read((char*)(&ucRead), 1);;  
    assert(ucRead == ucComma);  
    //next char is space  
    ifs.read((char*)(&ucRead), 1);;  
    assert(ucRead == ucSpace);  
  
    ifs.read((char*)(&ucRead), 1);  
    assert(ucRead == ucLeftbrace);  
    arrStack[++iTop] = 1;//iTop is 1  
    ifs.read((char*)(&ucRead), 1);  
    assert(ucRead == ucLeftbrace);  
    arrStack[++iTop] = 1;//iTop is 2  
  
    while (iTop > 1 && ifs.eof() == false)  
    {  
        readDoubleFromJson(ifs, vpdw, vdb, vecDigit, pdvt, iIndex, ihidden, arrStack, iTop, true);  
    };  
    //next char is dot  
    ifs.read((char*)(&ucRead), 1);;  
    assert(ucRead == ucComma);  
    cout << vpdw.size() << endl;  
    //next char is space  
    while (iTop > -1 && ifs.eof() == false)  
    {  
        readDoubleFromJson(ifs, vpdw, vdb, vecDigit, pdvt, iIndex, ihidden, arrStack, iTop, false);  
    };  
    assert(vpdw.size() == vdb.size());  
    assert(iTop == -1);  
    vvAllw.push_back(vpdw);  
    vvAllb.push_back(vdb);  
    //clear vpdw and pdb 's contents  
    vpdw.clear();  
    vdb.clear();  
    //close file  
    ifs.close();  
    return true;  
  
  
}  
  
bool LoadWeighFromJson(vector< vector<double*> > &vvAllw, vector< vector<double> > &vvAllb,  
                     const char *filename, const vector<int> &vecSecondDimOfWeigh)  
{  
    cout << "LoadWeighFromJson" << endl;  
    const int szistsize = 10;  
  
    const BYTE ucLeftbrace = '[', ucRightbrace = ']', ucComma = ',', ucSpace = ' ';  
    int arrStack[szistsize], iTop = -1, iIndex = 0;  
  
  
    ifstream ifs;  
    ifs.open(filename, ios::in | ios::binary);  
    assert(ifs.is_open());  
  
  
    double *pdvt;  
    BYTE ucRead;  
    ifs.read((char*)(&ucRead), 1);  
    assert(ucRead == ucLeftbrace);  
    //栈中全部存放左括号，用1代表,0说明清除  
    arrStack[++iTop] = 1;  
    ifs.read((char*)(&ucRead), 1);  
    assert(ucRead == ucLeftbrace);  
    arrStack[++iTop] = 1;//iTop is 1  
    ifs.read((char*)(&ucRead), 1);  
    assert(ucRead == ucLeftbrace);  
    arrStack[++iTop] = 1;//iTop is 2  
  
    int iVecWeighSize = vecSecondDimOfWeigh.size();  
    vector<BYTE> vecDigit;  
    vector<double *> vpdw;  
    vector<double> vdb;  
    //读取iVecWeighSize个[w,b]  
    for (int i = 0; i < iVecWeighSize; ++i)  
    {  
        int iDimesionOfWeigh = vecSecondDimOfWeigh[i];  
        while (iTop > 1 && ifs.eof() == false)  
        {  
            readDoubleFromJson(ifs, vpdw, vdb, vecDigit, pdvt, iIndex, iDimesionOfWeigh, arrStack, iTop, true);  
        };  
        //next char is dot  
        ifs.read((char*)(&ucRead), 1);;  
        assert(ucRead == ucComma);  
        cout << vpdw.size() << endl;  
        //next char is space  
        while (iTop > 0 && ifs.eof() == false)  
        {  
            readDoubleFromJson(ifs, vpdw, vdb, vecDigit, pdvt, iIndex, iDimesionOfWeigh, arrStack, iTop, false);  
        };  
        assert(vpdw.size() == vdb.size());  
        assert(iTop == 0);  
        vvAllw.push_back(vpdw);  
        vvAllb.push_back(vdb);  
        //clear vpdw and pdb 's contents  
        vpdw.clear();  
        vdb.clear();  
        //如果最后一对[w,b]读取完毕，就退出，下一个字符是右括号']'  
        if (i >= iVecWeighSize - 1)  
        {  
            break;  
        }  
  
        //next char is comma  
        ifs.read((char*)(&ucRead), 1);;  
        assert(ucRead == ucComma);  
        //next char is space  
        ifs.read((char*)(&ucRead), 1);;  
        assert(ucRead == ucSpace);  
  
        ifs.read((char*)(&ucRead), 1);  
        assert(ucRead == ucLeftbrace);  
        arrStack[++iTop] = 1;//iTop is 1  
        ifs.read((char*)(&ucRead), 1);  
        assert(ucRead == ucLeftbrace);  
        arrStack[++iTop] = 1;//iTop is 2  
    }  
  
      
    ifs.read((char*)(&ucRead), 1);;  
    assert(ucRead == ucRightbrace);  
    --iTop;  
    assert(iTop == -1);  
      
    //close file  
    ifs.close();  
    return true;  
  
      
}  
  
  
//read vt from mnszist, format is [[[], [],..., []],[1, 3, 5,..., 7]]  
bool LoadTestSampleFromJson(vector<double*> &vecTrain, vector<WORD> &vecLabel, const char *filename, const int m_iInput)  
{  
    cout << "LoadTestSampleFromJson" << endl;  
    const int szistsize = 10;  
    const int ciFeaturesize = m_iInput;  
    const BYTE ucLeftbrace = '[', ucRightbrace = ']', ucComma = ',', ucSpace = ' ';  
    int arrStack[szistsize], iTop = -1, iIndex = 0;  
  
  
    ifstream ifs;  
    ifs.open(filename, ios::in | ios::binary);  
    assert(ifs.is_open());  
  
  
    double *pdvt;  
    BYTE ucRead;  
    ifs.read((char*)(&ucRead), 1);  
    assert(ucRead == ucLeftbrace);  
    //栈中全部存放左括号，用1代表,0说明清除  
    arrStack[++iTop] = 1;  
    ifs.read((char*)(&ucRead), 1);  
    assert(ucRead == ucLeftbrace);  
    arrStack[++iTop] = 1;//iTop is 1  
    vector<BYTE> vecDigit;  
    while (iTop > 0 && ifs.eof() == false)  
    {  
        readDigitFromJson(ifs, vecTrain, vecLabel, vecDigit, pdvt, iIndex, ciFeaturesize, arrStack, iTop, true);  
    };  
    //next char is dot  
    ifs >> ucRead;  
    assert(ucRead == ucComma);  
    cout << vecTrain.size() << endl;  
    //next char is space  
//    ifs.read((char*)(&ucRead), 1);  
//    ifs.read((char*)(&ucRead), 1);  
//    assert(ucRead == ucLeftbrace);  
    while (iTop > -1 && ifs.eof() == false)  
    {  
        readDigitFromJson(ifs, vecTrain, vecLabel, vecDigit, pdvt, iIndex, ciFeaturesize, arrStack, iTop, false);  
    };  
    assert(vecLabel.size() == vecTrain.size());  
    assert(iTop == -1);  
  
    ifs.close();  
  
  
    return true;  
}  
void MakeOneLabel(int iMax, double *pdLabel, int m_iOut)  
{  
    for (int j = 0; j < m_iOut; ++j)  
        pdLabel[j] = 0;  
    pdLabel[iMax] = 1.0;  
}  
  
void MakeCnnSample(double arrInput[2][64], double *pdImage, int iImageWidth, int iNumOfImage)  
{  
    int iImageSize = iImageWidth * iImageWidth;  
  
    for (int k = 0; k < iNumOfImage; ++k)  
    {  
        int iStart = k *iImageSize;  
        for (int i = 0; i < iImageWidth; ++i)  
        {  
            for (int j = 0; j < iImageWidth; ++j)  
            {  
                int iIndex = iStart + i * iImageWidth + j;  
                pdImage[iIndex] = 1;  
                pdImage[iIndex] += i + j;  
                if (k > 0)  
                    pdImage[iIndex] -= 1;  
                arrInput[k][i * iImageWidth +j] = pdImage[iIndex];  
                //pdImage[iIndex] /= 15.0   ;  
  
            }  
        }  
    }  
      
    cout << "input image is\n";  
    for (int k = 0; k < iNumOfImage; ++k)  
    {  
        int iStart = k *iImageSize;  
        cout << "k is " << k <<endl;  
        for (int i = 0; i < iImageWidth; ++i)  
        {  
            for (int j = 0; j < iImageWidth; ++j)  
            {  
                int iIndex =  i * iImageWidth + j;  
                double dValue = arrInput[k][iIndex];  
                cout << dValue << ' ';  
  
            }  
            cout << endl;  
  
        }  
        cout << endl;  
  
    }  
      
    cout << endl;  
}  
  
void MakeCnnWeigh(double *pdKernel, int iNumOfKernel)  
{  
    const int iKernelWidth = 3;  
    double iSum = 0;  
    double arrKernel[iKernelWidth][iKernelWidth] = {{4, 7, 1},  
                        {3, 8, 5},  
                        {3, 2, 3}};  
    double arr2[iKernelWidth][iKernelWidth] = {{6, 5, 4},  
                                                {5, 4, 3},  
                                                {4, 3, 2}};  
    for (int k = 0; k < iNumOfKernel; ++k)  
    {  
        int iStart = k * iKernelWidth * iKernelWidth;  
        for (int i = 0; i < iKernelWidth; ++i)  
        {  
            for (int j = 0; j < iKernelWidth; ++j)  
            {  
                int iIndex = i * iKernelWidth + j + iStart;  
                pdKernel[iIndex] = i + j + 2;  
                if (k > 0)  
                    pdKernel[iIndex] = arrKernel[i][j];  
                iSum += pdKernel[iIndex];  
  
            }  
        }  
    }  
    cout << "sum is " << iSum << endl;  
    for (int k = 0; k < iNumOfKernel; ++k)  
    {  
        cout << "kernel :" << k << endl;  
        int iStart = k * iKernelWidth * iKernelWidth;  
        for (int i = 0; i < iKernelWidth; ++i)  
        {  
            for (int j = 0; j < iKernelWidth; ++j)  
            {  
                int iIndex = i * iKernelWidth + j + iStart;  
                //pdKernel[iIndex] /= (double)iSum;  
                cout << pdKernel[iIndex] << ' ';  
  
  
            }  
            cout << endl;  
  
        }  
        cout << endl;  
    }  
  
      
   cout << endl;  
}  
