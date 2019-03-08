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
void initTensorZero(T* data, size_t size){
    std::fill(&data[0], &data[0] + size, 0);
}

//----------------------------------------------------------------------
// kernels for OpNode kernels

#define A(i, j) a[i*lda + j]
#define B(i, j) b[i*ldb + j]
#define C(i, j) c[i*ldc + j]

//TODO use template?

void matrixMatrixMul_f(int m, int n, int k, const float *a, int lda,
                       const float *b, int ldb, float *c, int ldc) {
    // The order of these loops is tuned for column-major matrices.
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C(i, j) = 0.f;
        }
    }
  for (int p = 0; p < k; p++) {
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        C(i, j) += A(i, p) * B(p, j);
      }
    }
  }
}

void matrixTanh_f(int m, int n, const float *a, int lda, float *b, int ldb){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            B(i, j) = 1 - 2 / (expf(A(i, j) * 2) + 1);
        }
    }
}


// (m, n): dims of a
// (n, m): dims of b
void matrixTrans_f(int m, int n, const float *a, int lda, float *b, int ldb){
    for(int i=0; i<n; i++){
        for(int j=0; j<m; j++){
            B(i, j) = A(j, i); 
        }
    }
}

// a: grad of original input
// b: original output
// c: grad of original output
void matrixTanhGrad_f(int m, int n, float *a, int lda, 
                      const float *b, int ldb, const float *c, int ldc){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            A(i, j) = (1 - B(i, j)*B(i, j)) * C(i, j);
        }
    }
}

void matrixSoftmax_f(int m, int n, const float *a, int lda, float *b, int ldb) {
    for(int i=0; i<m; i++){
        float max_ = A(i,0);
        for(int j=0; j<n; j++){
            max_ = std::max(max_, A(i,j));
        }
        float sum = 0;
        for(int j=0; j<n; j++){
            B(i,j) = expf(A(i,j) - max_);
            sum += B(i,j);
        }
        for(int j=0; j<n; j++){
            B(i,j) = B(i,j) / sum;
        }
    }
}

// a: grad of input
// b: original out
// selected: selected
// 比如输出为[0.1, 0.6, 0.3]正确答案为1, 那么梯度就是[0.1, -0.4, 0.3]
void matrixSoftmaxGrad_f(int m, int n, float *a, int lda, 
                          const float *b, int ldb, const int *selected) {
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            float delta = (selected[i] == j);
            A(i, j) = B(i, j) - delta;
        }
    }
}

void vecAdd_f(int size, float *a, float *b, float *c){
    for(int i=0; i<size; i++){
        c[i] = a[i] + b[i];
    }    
}

void printMatrix_f(int m, int n, const float *a, int lda){
    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            std::cout << A(i, j) << " ";
        }
        std::cout << std::endl;
    }
}
)";
