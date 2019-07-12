/*************************************************************************
        > File Name: kernels.h
        > Author: wayne
        > Mail:
        > Created Time: 二  1/22 19:06:41 2019
 ************************************************************************/
// static const char *KERNELS_CODE = R"(
//----------------------------------------------------------------------
// kernels for tensor init
template <typename T>
void initTensorXavier(T *data, size_t size, float filterSize) {
    std::random_device rd;
    // std::mt19937 engine_(std::mt19937::default_seed);
    std::mt19937 engine_(rd());
    double scale = std::sqrt(3.0 / double(filterSize));
    std::uniform_real_distribution<> values(-scale, scale);
    for (size_t i = 0; i < size; i++) {
        data[i] = values(engine_);
    }
}

template <typename T> void initTensorConstant(T *data, size_t size, float value) {
    std::fill(&data[0], &data[0] + size, value);
}

template <typename T> void initTensorZero(T *data, size_t size) {
    std::fill(&data[0], &data[0] + size, 0);
}

//----------------------------------------------------------------------
// kernels for OpNode kernels

#define A(i, j) a[i * lda + j]
#define B(i, j) b[i * ldb + j]
#define C(i, j) c[i * ldc + j]
#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

// TODO use template?

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

void matrixTanh_f(int m, int n, const float *a, int lda, float *b, int ldb) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            B(i, j) = 1 - 2 / (expf(A(i, j) * 2) + 1);
        }
    }
}

// (m, n): dims of a
// (n, m): dims of b
void matrixTrans_f(int m, int n, const float *a, int lda, float *b, int ldb) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            B(i, j) = A(j, i);
        }
    }
}

// a: grad of original input
// b: original output
// c: grad of original output
void matrixTanhGrad_f(int m, int n, float *a, int lda, const float *b, int ldb,
                      const float *c, int ldc) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            A(i, j) = (1 - B(i, j) * B(i, j)) * C(i, j);
        }
    }
}

void matrixSoftmax_f(int m, int n, const float *a, int lda, float *b, int ldb) {
    for (int i = 0; i < m; i++) {
        float max_ = A(i, 0);
        for (int j = 0; j < n; j++) {
            max_ = std::max(max_, A(i, j));
        }
        float sum = 0;
        for (int j = 0; j < n; j++) {
            B(i, j) = expf(A(i, j) - max_);
            sum += B(i, j);
        }
        for (int j = 0; j < n; j++) {
            B(i, j) = B(i, j) / sum;
        }
    }
}

void matrixSoftmaxWithLoss_f(int m, int n, const float *a, int lda, float *b, int ldb, const int *selected, float* loss) {
    *loss = 0;
    for (int i = 0; i < m; i++) {
        float max_ = A(i, 0);
        for (int j = 0; j < n; j++) {
            max_ = std::max(max_, A(i, j));
        }
        float sum = 0;
        for (int j = 0; j < n; j++) {
            B(i, j) = expf(A(i, j) - max_);
            sum += B(i, j);
        }
        for (int j = 0; j < n; j++) {
            B(i, j) = B(i, j) / sum;
        }

        int k = selected[i];
        *loss += logf(sum * expf(max_)) - A(i, k);
    }
    //loss https://blog.csdn.net/Iriving_shu/article/details/78609409
    // log(sigma(e^z_i)) - z_k (k-selected)
    *loss /= m;
}

// a: grad of input
// b: original out
// selected: selected
// 比如输出为[0.1, 0.6, 0.3]正确答案为1, 那么梯度就是[0.1, -0.4, 0.3]
void matrixSoftmaxGrad_f(int m, int n, float *a, int lda, const float *b,
                         int ldb, const int *selected) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = (selected[i] == j);
            A(i, j) = B(i, j) - delta;
        }
    }
}

void matrixSoftmaxWithLossGrad_f(int m, int n, float *a, int lda, const float *b,
                         int ldb, const int *selected) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float delta = (selected[i] == j);
            A(i, j) = B(i, j) - delta;
        }
    }
}

/// \returns the index of the element at x,y,z,w.
inline size_t getIdx4d(const size_t *dims, size_t x, size_t y, size_t z,
                       size_t w) {
    return (x * dims[1] * dims[2] * dims[3]) + (y * dims[2] * dims[3]) +
           (z * dims[3]) + w;
}
inline size_t getIdx2d(const size_t *dims, size_t x, size_t y) {
    return (x * dims[1] + y);
}

void conv2d_set_bias(size_t n, float *output, const float *bias,
                     const size_t *odims, const size_t *bdims) {

    for (size_t ox = 0; ox < odims[1]; ox++) {
        for (size_t oy = 0; oy < odims[2]; oy++) {
            for (size_t co = 0; co < odims[3]; co++) {
                float value = bias[co];
                auto idx = getIdx4d(odims, n, ox, oy, co);
                output[idx] = value;
            } // Co
        }     // W
    }         // H
}

void conv2d_f(float *output, const float *input, const float *filter,
              const float *bias, const size_t *odims, const size_t *idims,
              const size_t *fdims, const size_t *bdims, const size_t *kernels,
              const size_t *strides, const size_t *pads, const size_t group) {
    size_t batch = idims[0];
    size_t inC = idims[3];
    size_t oC = odims[3];
    size_t inCG = inC / group;
    size_t oCG = oC / group;

    size_t kernel_h = kernels[0];
    size_t kernel_w = kernels[1];
    size_t stride_h = strides[0];
    size_t stride_w = strides[1];
    size_t pad_t = pads[0];
    size_t pad_l = pads[1];

    // for each img
    for (size_t n = 0; n < batch; n++) {

        conv2d_set_bias(n, output, bias, odims, bdims);

        // for each group
        for (size_t g = 0; g < group; g++) {

            for (size_t co = g * oCG; co < (g + 1) * oCG; co++) {

                for (size_t kx = 0; kx < kernel_h; kx++) {
                    for (size_t ky = 0; ky < kernel_w; ky++) {

                        // filter idx
                        size_t fIdx_base =
                            getIdx4d(fdims, co, kx, ky, g * inCG);

                        // for each output (x, y)
                        for (size_t ox = 0; ox < odims[1]; ox++) {
                            for (size_t oy = 0; oy < odims[2]; oy++) {

                                ssize_t ix =
                                    (ssize_t)ox * stride_h - pad_t + kx;
                                ssize_t iy =
                                    (ssize_t)oy * stride_w - pad_l + ky;

                                if (ix < 0 || ix >= (ssize_t)idims[1] ||
                                    iy < 0 || iy >= (ssize_t)idims[2]) {
                                    continue;
                                }

                                // input idx
                                size_t iIdx_base =
                                    getIdx4d(idims, n, ix, iy, g * inCG);
                                // output idx
                                size_t oIdx = getIdx4d(odims, n, ox, oy, co);

                                // for each input channel per group
                                for (size_t ci = 0; ci < inCG; ci++) {
                                    output[oIdx] += input[iIdx_base + ci] *
                                                    filter[fIdx_base + ci];
                                } // W
                            }     // H
                        }
                    }
                } //
            }     // Co
        }         // group
    }             // N
}

void conv2dGrad_f(float *inputG, float *filterG, float *biasG,
                const float* outputG, const float *input, const float *filter,
              const size_t *odims, const size_t *idims,
              const size_t *fdims, const size_t *bdims, const size_t *kernels,
              const size_t *strides, const size_t *pads, const size_t group) {
    size_t batch = idims[0];
    size_t inC = idims[3];
    size_t oC = odims[3];
    size_t inCG = inC / group;
    size_t oCG = oC / group;

    size_t kernel_h = kernels[0];
    size_t kernel_w = kernels[1];
    size_t stride_h = strides[0];
    size_t stride_w = strides[1];
    size_t pad_t = pads[0];
    size_t pad_l = pads[1];

    initTensorZero(inputG, idims[0] * idims[1] * idims[2] * idims[3]);
    initTensorZero(filterG, fdims[0] * fdims[1] * fdims[2] * fdims[3]);
    initTensorZero(biasG, oC);

    // for each img
    for (size_t n = 0; n < batch; n++) {


        // for each group
        for (size_t g = 0; g < group; g++) {

            for (size_t co = g * oCG; co < (g + 1) * oCG; co++) {

                for (size_t kx = 0; kx < kernel_h; kx++) {
                    for (size_t ky = 0; ky < kernel_w; ky++) {

                        // filter idx
                        size_t fIdx_base =
                            getIdx4d(fdims, co, kx, ky, g * inCG);

                        // for each output (x, y)
                        for (size_t ox = 0; ox < odims[1]; ox++) {
                            for (size_t oy = 0; oy < odims[2]; oy++) {

                                ssize_t ix =
                                    (ssize_t)ox * stride_h - pad_t + kx;
                                ssize_t iy =
                                    (ssize_t)oy * stride_w - pad_l + ky;

                                if (ix < 0 || ix >= (ssize_t)idims[1] ||
                                    iy < 0 || iy >= (ssize_t)idims[2]) {
                                    continue;
                                }

                                // input idx
                                size_t iIdx_base =
                                    getIdx4d(idims, n, ix, iy, g * inCG);
                                // output idx
                                size_t oIdx = getIdx4d(odims, n, ox, oy, co);

                                // for each input channel per group
                                for (size_t ci = 0; ci < inCG; ci++) {
                                    inputG[iIdx_base + ci] += filter[fIdx_base + ci] * outputG[oIdx];
                                    filterG[fIdx_base + ci] += input[iIdx_base + ci] * outputG[oIdx];
                                }

                                if(kx==0 && ky==0) {
                                    biasG[co] += outputG[oIdx];
                                }
                            } // W
                        }     // H
                    }   //kw
                } // kh
            }     // Co
        }         // group
    }             // N
}



void maxpool_f(const float *input, float *output, const size_t *idims,
               const size_t *odims, const size_t *kernels,
               const size_t *strides, const size_t *pads) {
    size_t kernel_h = kernels[0];
    size_t kernel_w = kernels[1];
    size_t stride_h = strides[0];
    size_t stride_w = strides[1];
    size_t pad_t = pads[0];
    size_t pad_l = pads[1];

    // for each img in batch
    for (size_t n = 0; n < idims[0]; n++) {
        ssize_t ix_b = -(ssize_t)pad_t;

        // for each output (x, y)
        for (size_t ox = 0; ox < odims[1]; ix_b += stride_h, ox++) {

            ssize_t iy_b = -(ssize_t)pad_l;
            for (size_t oy = 0; oy < odims[2]; iy_b += stride_w, oy++) {

                // for each channel
                for (size_t c = 0; c < idims[3]; c++) {

                    size_t oIdx = getIdx4d(odims, n, ox, oy, c);
                    bool uninit = true;
                    float max = 0;

                    // for each filter (x,y)
                    for (size_t kx = 0; kx < kernel_h; kx++) {
                        for (size_t ky = 0; ky < kernel_w; ky++) {
                            ssize_t ix = ix_b + kx;
                            ssize_t iy = iy_b + ky;

                            if (ix < 0 || ix >= (ssize_t)idims[1] || iy < 0 ||
                                iy >= (ssize_t)idims[2]) {
                                continue;
                            }

                            size_t iIdx = getIdx4d(idims, n, ix, iy, c);
                            float value = input[iIdx];
                            if (uninit || (value > max)) {
                                max = value;
                                uninit = false;
                            }
                        }

                    }

                    output[oIdx] = max;

                } // C
            } // W
        }     // H
    }         // N
}

// 未记录maxpool输出对应原始输入的xy，因此必不可少重复maxpool的流程
void maxpoolGrad_f(float *inputG, const float *outputG,
               const float *input, const size_t *idims,/*idims = iGdims*/
               const size_t *odims, const size_t *kernels,
               const size_t *strides, const size_t *pads) {
    size_t kernel_h = kernels[0];
    size_t kernel_w = kernels[1];
    size_t stride_h = strides[0];
    size_t stride_w = strides[1];
    size_t pad_t = pads[0];
    size_t pad_l = pads[1];

    initTensorZero(inputG, idims[0] * idims[1] * idims[2] * idims[3]);

    // for each img in batch
    for (size_t n = 0; n < idims[0]; n++) {
        ssize_t ix_b = -(ssize_t)pad_t;

        // for each output (x, y)
        for (size_t ox = 0; ox < odims[1]; ix_b += stride_h, ox++) {

            ssize_t iy_b = -(ssize_t)pad_l;
            for (size_t oy = 0; oy < odims[2]; iy_b += stride_w, oy++) {

                // for each channel
                for (size_t c = 0; c < idims[3]; c++) {

                    size_t oIdx = getIdx4d(odims, n, ox, oy, c);
                    bool uninit = true;
                    float max = 0;
                    size_t max_ix = 0;
                    size_t max_iy = 0;

                    // for each filter (x,y)
                    for (size_t kx = 0; kx < kernel_h; kx++) {
                        for (size_t ky = 0; ky < kernel_w; ky++) {
                            ssize_t ix = ix_b + kx;
                            ssize_t iy = iy_b + ky;

                            if (ix < 0 || ix >= (ssize_t)idims[1] || iy < 0 ||
                                iy >= (ssize_t)idims[2]) {
                                continue;
                            }

                            size_t iIdx = getIdx4d(idims, n, ix, iy, c);
                            float value = input[iIdx];
                            if (uninit || (value > max)) {
                                max = value;
                                max_ix = ix;
                                max_iy = iy;
                                uninit = false;
                            }
                        }
                        // output[oIdx] = max;
                    }

                    size_t iIdx = getIdx4d(idims, n, max_ix, max_iy, c);
                    inputG[iIdx] += outputG[oIdx];

                } // C
            } // W
        }     // H
    }         // N
}

void avgpool_f(const float *input, float *output, const size_t *idims,
               const size_t *odims, const size_t *kernels,
               const size_t *strides, const size_t *pads) {
    size_t kernel_h = kernels[0];
    size_t kernel_w = kernels[1];
    float kernel_cnt = kernel_h * kernel_w;
    size_t stride_h = strides[0];
    size_t stride_w = strides[1];
    size_t pad_t = pads[0];
    size_t pad_l = pads[1];

    // for each img in batch
    for (size_t n = 0; n < idims[0]; n++) {
        ssize_t ix_b = -(ssize_t)pad_t;

        // for each output (x, y)
        for (size_t ox = 0; ox < odims[1]; ix_b += stride_h, ox++) {

            ssize_t iy_b = -(ssize_t)pad_l;
            for (size_t oy = 0; oy < odims[2]; iy_b += stride_w, oy++) {

                // for each channel
                for (size_t c = 0; c < idims[3]; c++) {

                    size_t oIdx = getIdx4d(odims, n, ox, oy, c);

                    float sum = 0;

                    // for each filter (x,y)
                    for (size_t kx = 0; kx < kernel_h; kx++) {
                        for (size_t ky = 0; ky < kernel_w; ky++) {
                            ssize_t ix = ix_b + kx;
                            ssize_t iy = iy_b + ky;

                            if (ix < 0 || ix >= (ssize_t)idims[1] || iy < 0 ||
                                iy >= (ssize_t)idims[2]) {
                                continue;
                            }

                            size_t iIdx = getIdx4d(idims, n, ix, iy, c);
                            sum += input[iIdx];
                        }

                        output[oIdx] = sum / kernel_cnt;
                    } // C
                }
            } // W
        }     // H
    }         // N
}

// NHWC
void batchnormalization_f(float *output, const float *input, const float *mean,
                          const float *var, const float *scale,
                          const float *bias, const size_t *idims,
                          const float epsilon) {
    for (size_t n = 0; n < idims[0]; n++) {

        for (size_t h = 0; h < idims[1]; h++) {

            for (size_t w = 0; w < idims[2]; w++) {

                for (size_t c = 0; c < idims[3]; c++) {

                    size_t idx = getIdx4d(idims, n, h, w, c);
                    output[idx] = (input[idx] - mean[c]) *
                                      std::pow((var[c] + epsilon), -0.5) *
                                      scale[c] +
                                  bias[c];
                }
            }
        }
    }
}

void transpose4d_f(const float *input, float *output, const size_t *idims,
                   const size_t *odims, const size_t *shuffle) {
    // For each layer in the batch:
    // assert 4d
    size_t idx[4];
    for (idx[0] = 0; idx[0] < idims[0]; idx[0]++)
        for (idx[1] = 0; idx[1] < idims[1]; idx[1]++)
            for (idx[2] = 0; idx[2] < idims[2]; idx[2]++)
                for (idx[3] = 0; idx[3] < idims[3]; idx[3]++)
                    output[getIdx4d(odims, idx[shuffle[0]], idx[shuffle[1]],
                                    idx[shuffle[2]], idx[shuffle[3]])] =
                        input[getIdx4d(idims, idx[0], idx[1], idx[2], idx[3])];
}

void transpose2d_f(const float *input, float *output, const size_t *idims,
                   const size_t *odims, const size_t *shuffle) {
    size_t idx[2];
    for (idx[0] = 0; idx[0] < idims[0]; idx[0]++)
        for (idx[1] = 0; idx[1] < idims[1]; idx[1]++)
            output[getIdx2d(odims, idx[shuffle[0]], idx[shuffle[1]])] =
                input[getIdx2d(idims, idx[0], idx[1])];
}

void batchedadd_f(float *dest, const float *batch, const float *slice,
                  size_t numSlice, size_t sliceSize) {
    // For each layer in the batch:
    for (size_t n = 0; n < numSlice; n++) {
        size_t base = n * sliceSize;
        // For each element in the slice.
        for (size_t i = 0; i < sliceSize; i++) {
            dest[base + i] = batch[base + i] + slice[i];
        }
    }
}

void batchedreduceadd_f(float *dest, const float *batch, size_t numSlice, size_t sliceSize) {
    for (size_t n = 0; n < numSlice; n++) {
        float sum = 0;
        size_t base = n * sliceSize;
        for (size_t i = 0; i < sliceSize; i++) {
            sum += batch[base+i];
        }
        dest[n] = sum;
    }
}

void sgd_f(size_t size, float *out, float *w, float *dw, float *dw_mom, float lr, float decay, float momentum, size_t batch) {
    float neglr = -lr / batch;
    for(int i=0; i<size; i++){

        float negDelta = (w[i]*decay + dw[i]) * neglr + dw_mom[i]*momentum;
        dw_mom[i] = negDelta;
        out[i] += negDelta;
    }
}

void relu_f(const float *src, float *dest, size_t size) {
    // For each layer in the batch:
    for (size_t idx = 0; idx < size; idx++) {
        dest[idx] = (src[idx] > 0) ? src[idx] : 0;
    }
}

// srcGrad: grad of original input
// src: original input
// destGrad: grad of original output
// elementwise, if value of src[:] > 0, let destGrad pass, or 0
void reluGrad_f(float *srcGrad, const float *src, const float *destGrad, size_t size) {
    for(size_t i=0; i<size; i++) {
        srcGrad[i] = (src[i] > 0) ? destGrad[i] : 0;
    }
}

void vecAdd_f(int size, float *a, float *b, float *c) {
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

/**
* reference: Caffe
*/
void argMax_f(const float *input, int *idx, int m, int n, int top_k) {
    for(int i=0; i<m; i++) {
        std::vector<std::pair<float, int>> value_idx(n);
        for(int j=0; j<n; j++) {
            value_idx[j] = std::make_pair(input[i*n+j], j);
        }
        std::partial_sort(value_idx.begin(), value_idx.begin()+top_k, value_idx.end(), std::greater<std::pair<float, int>>());

        for(int j=0; j<top_k; j++) {
            idx[i*top_k+j] = value_idx[j].second;
        }
    }
}

template <typename T>
void printMatrix(const T *a, int m, int n) {
    std::cout.flags(std::ios::fixed);
    std::cout.precision(3);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << a[i*n+j] << " ";
        }
        std::cout << std::endl;
    }
}
// )";
