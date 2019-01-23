/*************************************************************************
	> File Name: kernels.h
	> Author: wayne
	> Mail: singleon11@gmail.com 
	> Created Time: 二  1/22 19:06:41 2019
 ************************************************************************/ 
static const char *KERNELS_CODE = R"(
//----------------------------------------------------------------------
// kernels for tensor init
template<typename T>
void initTensorXavier(T* data, size_t size, float filterSize){
    std::random_device rd;     
    // std::mt19937 engine_(std::mt19937::default_seed);
    std::mt19937 engine_(rd());
    double scale  = std::sqrt(3.0 / double(filterSize));
    std::uniform_real_distribution<> values(-scale, scale); 
    for(size_t i=0; i<size; i++){
        data[i] = values(engine_);
    }
}

template<typename T>
void initTensorConstant(T* data, size_t size, T value){
    std::fill(&data[0], &data[0] + size, value);
}

template<typename T>
void initTensorZero(T* data, size_t size, T value){
    std::fill(&data[0], &data[0] + size, value);
}

//----------------------------------------------------------------------
// kernels for OpNode kernels

#define A(i, j) a[(j)*lda + (i)]
#define B(i, j) b[(j)*ldb + (i)]
#define C(i, j) c[(j)*ldc + (i)]

//TODO use template?

void matrixMatrixMul_f(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc) {
  // The order of these loops is tuned for column-major matrices.
  for (int p = 0; p < k; p++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        C(i, j) += A(i, p) * B(p, j);
      }
    }
  }
}
void matrixMatrixMul_d(int m, int n, int k, const double *a, int lda,
                       const double *b, int ldb, double *c, int ldc) {
  // The order of these loops is tuned for column-major matrices.
  for (int p = 0; p < k; p++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        C(i, j) += A(i, p) * B(p, j);
      }
    }
  }
}


void matrixTanh_f(int m, int n, float *a, int lda, float *b, int ldb){
  for(int j=0; j<n; j++){
    for(int i=0; i<m; i++){
      B(i, j) = 1 - 2 / expf((A(i, j) * 2) + 1);
    }
  }
}
void matrixTanh_d(int m, int n, double *a, int lda, double *b, int ldb){
  for(int j=0; j<n; j++){
    for(int i=0; i<m; i++){
      B(i, j) = 1 - 2 / expf((A(i, j) * 2) + 1);
    }
  }
}

void matrixSoftmax_f(const float *inW, float *outW, const size_t *idim,
                      const size_t *odim) {
  // unimplemented!
}
)";
