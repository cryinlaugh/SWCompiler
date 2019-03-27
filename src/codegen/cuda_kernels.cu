static const char *CUDA_CODE = R"(
//----------------------------------------------------------------------
// cuda kernels for OpNode
__global__ void matrixSoftmax_float(float *src, float *dest,
                            int sliceSize) {
  int i = blockIdx.x *blockDim.x + threadIdx.x;
  float max_ = src[i * sliceSize];
  for (int j = 0; j < sliceSize; j++) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) {
    float e = exp(src[i * sliceSize + j] - max_);
    sum += e;
    dest[i * sliceSize + j] = e;
  }
  for (int j = 0; j < sliceSize; j++) {
    dest[i * sliceSize + j] /= sum;
  }
}

__global__ void matrixTanh_float(float *src, float *dest, int n){
    int i = blockIdx.x *blockDim.x + threadIdx.x;
    for(int j=0; j<n; j++){
        dest[i*n + j] = 1 - 2 / (expf(src[i*n + j] * 2) + 1);
    }
}
)";