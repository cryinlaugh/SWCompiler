/*************************************************************************
	> File Name: image.h
	> Author: wayne
	> Mail: singleon11@gmail.com 
	> Created Time: 四  1/24 15:15:53 2019
 ************************************************************************/

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>

unsigned load(float *data, size_t size, size_t offset, std::string inputFile) {
  
    std::ifstream input(inputFile, std::ios::binary);
  
    if (!input.is_open()) {
        std::cout << "Error loading " << inputFile << "\n";
        std::exit(EXIT_FAILURE);
    }
  
    std::vector<char> binData((std::istreambuf_iterator<char>(input)),
                           (std::istreambuf_iterator<char>()));
    float *dataAsFloatPtr = reinterpret_cast<float *>(&binData[0]);

    dataAsFloatPtr += offset;
    for(int i=0; i<size; i++){
        data[i] = dataAsFloatPtr[i];
    }
  
    return size;
}

template<typename T>
void store(T *data, size_t size,  std::string outFile) {
  
    std::ofstream output(outFile, std::ios::binary);
  
    if (!output.is_open()) {
        std::cout << "Error opening " << outFile<< "\n";
        std::exit(EXIT_FAILURE);
    }

    char *dataAsCharPtr  = reinterpret_cast<char *>(data);
    std::vector<char> tmpVector(dataAsCharPtr, dataAsCharPtr + sizeof(T)*size);
    std::copy(tmpVector.begin(), tmpVector.end(), std::ostreambuf_iterator<char>(output));
}

template<typename T>
unsigned loadLabel(T *data, size_t size, size_t offset, std::string inputFile) {
  
    std::ifstream input(inputFile, std::ios::binary);
  
    if (!input.is_open()) {
        std::cout << "Error loading " << inputFile << "\n";
        std::exit(EXIT_FAILURE);
    }
  
    std::vector<char> binData((std::istreambuf_iterator<char>(input)),
                           (std::istreambuf_iterator<char>()));
    char *dataAsCharPtr = reinterpret_cast<char *>(&binData[0]);

    dataAsCharPtr += offset;

    for(int i=0; i<size; i++){
        data [i] = dataAsCharPtr[i];
    }
  
    return size;
}

template <class ElemTy> static char valueToChar(ElemTy val) {
  char ch = ' ';
  if (val > 0.2) {
    ch = '.';
  }
  if (val > 0.4) {
    ch = ',';
  }
  if (val > 0.6) {
    ch = ':';
  }
  if (val > 0.8) {
    ch = 'o';
  }
  if (val > 1.0) {
    ch = 'O';
  }
  if (val > 1.5) {
    ch = '0';
  }
  if (val > 2.0) {
    ch = '@';
  }
  if (val < -0.1) {
    ch = '-';
  }
  if (val < -0.2) {
    ch = '~';
  }
  if (val < -0.4) {
    ch = '=';
  }
  if (val < -1.0) {
    ch = '#';
  }
  return ch;
}

void dumpAscii(float* data, int h, int w){
    
    for (size_t x = 0; x < h; x++) {
        for (size_t y = 0; y < w; y++) {
          float val = data[x*w + y];
          std::cout << valueToChar(val);
        }
        std::cout << "\n";
    }
}
