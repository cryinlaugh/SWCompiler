/*************************************************************************
	> File Name: common.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 04 Dec 2018 08:09:21 AM UTC
 ************************************************************************/

#ifndef _COMMON_H
#define _COMMON_H

#include <memory>
#include <vector>
#include <string>
#include <map>
#include <climits>

enum class DataType { Float_t, Double_t, Int8_t, Int32_t };

// TODO: remove SLICE, TILING and Use case
enum ParallelStrategy { 
    SLICE, 
    TILING,  
    MEM_SAVING,
    COMM_SAVING
};

enum BytesProto {
    ONE_BYTE_AS_INT,
    FOUR_BYTES_AS_FLOAT
};


struct TrainingConfig {
    std::string optimizer;
    float lr{0.001};
    float decay{0.001};
    float momentum{0.9};
    size_t batch{1};
    // snapshot interval iter
    size_t snapshot{0};
    // For Dataload
    size_t max_epoch{1};
    size_t max_iters{0};
    // interval for Graph output display
    size_t display{0};

    BytesProto label_bytes{ONE_BYTE_AS_INT};
    BytesProto data_bytes{FOUR_BYTES_AS_FLOAT};
    std::string train_data_file;
    size_t train_data_samples{0};

};

struct Config {
    bool train_mode{false};

    // cpu codegen config
    bool mkldnn{false};

    // Nvidia cuda codegen config
    bool cuda{false};
    bool cublas{false};
    bool cuda_stream{false};

    bool mpi{false};
    int mpi_size{1};

    TrainingConfig train_config;

    // dumplicate member as train
    bool use_dataloader{false};
    std::string dataloader_src;
    BytesProto label_bytes{ONE_BYTE_AS_INT};
    BytesProto data_bytes{FOUR_BYTES_AS_FLOAT};
    size_t dataloader_samples{0};
    size_t display{0};


    // for parallel strategy selection
    ParallelStrategy parallel_preference{MEM_SAVING}; 
    bool force_data_parallel{false};

    bool enable_lowering{true};

    // comment compute function calls to get pure communication time 
    bool compute_op_annotation{false};
    bool comm_op_annotation{false};

    // if true, annotate dataloader and (TBD ?emitTensorInitializations)
    bool benchmark{false};
};

enum OpType { TENSOR_OP, BASIC_OP, DL_OP };

enum NodeType { TENSOR_NODE, OP_NODE };

enum TensorType {
    D5 = 5,
    D4 = 4,
    D3 = 3,
    D2 = 2,
    D1 = 1,
    D0 = 0,
    UNKNOWN = -1
};

typedef enum {
    layout_default = 0,

    // for tensors
    layout_nchw,
    layout_nhwc,
    layout_nc,
    layout_cn
} mem_layout_t;
const std::map<int, std::string> MEM_LAYOUT = {{layout_default, "default"},
    {layout_nchw, "nchw"},
    {layout_nhwc, "nhwc"},
    {layout_nc, "nc"},
    {layout_cn, "cn"},
};

const std::map<std::string, std::string> dtype_mkldnn_datatype_map = {
    {"float", "memory::data_type::f32"},
    {"int", "memory::data_type::s32"}
};

const std::map<std::string, std::string> layout_mkldnn_format_tag_map = {
    {"nhwc", "memory::format_tag::nhwc"},
    {"nchw", "memory::format_tag::nchw"},
    {"nc", "memory::format_tag::nc"},
    {"cn", "memory::format_tag::cn"},
    {"x", "memory::format_tag::x"},
    {"xy", "memory::format_tag::nc"},
};

enum class TensorInitType { NONE, CONSTANT, ZERO, XAVIER, FILE, PARENTOP };
enum class DeviceType : int { CPU, GPU };

enum class PrintStreamType { COUT, FILE };

struct Device {
    int rank{0};
    DeviceType type;
    int id{0};
    Device(int r = 0, DeviceType t = DeviceType::CPU, int i = 0) : rank{r}, type(t), id(i) {}
    friend bool operator==(const Device &x, const Device &y) {
        return x.rank == y.rank && x.type == y.type && x.id == y.id;
    }
};
namespace std {
template <> struct hash<Device> {
    size_t operator()(const Device &d) const {
        auto h0 = std::hash<int>{}(d.rank);
        auto h1 = std::hash<int>{}(static_cast<int>(d.type));
        auto h2 = std::hash<int>{}(d.id);
        return h0 ^ h1 ^ h2;
    }
};
} // namespace std

#define NCHW2NHWC                                                              \
    { 0, 2, 3, 1 }
#define NHWC2NCHW                                                              \
    { 0, 3, 1, 2 }

// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname)                                           \
    char gInstantiationGuard##classname;                                       \
    template class classname<float>;                                           \
    template class classname<double>

template <typename Dtype> class SWMem {
  private:
    size_t _len;
    Dtype *_data;

  public:
    SWMem(size_t len, Dtype *data);
    ~SWMem();
    Dtype *data();
    Dtype *mutable_data();
};

template <typename U, typename V>
int delVecMember(std::vector<U> &vec, V &del) {
    int delDone = 0;
    for (typename std::vector<U>::iterator it = vec.begin(); it != vec.end();) {
        if (*it == del) {
            it = vec.erase(it);
            delDone = 1;
            break;
        } else {
            ++it;
        }
    }
    return delDone;
}

#endif
