/*************************************************************************
    > File Name: DataLoader.h
    > Author: wayne
    > Mail:  
    > Created Time: 四  5/16 20:37:26 2019
 ************************************************************************/
#include <vector>
#include <string>
#include <cstdio>
#include <fstream>

enum BytesProto {
    ONE_BYTE_AS_INT,
    FOUR_BYTES_AS_FLOAT
};

class DataLoader {
    //FILE *_file{nullptr};
    std::ifstream _stream;
    size_t _sample_num{0};

    std::vector<size_t> _label_batch_shape; 
    std::vector<size_t> _data_batch_shape; 

    size_t _label_bytes{1}; // one label
    size_t _data_bytes{1}; // one sample
    size_t _data_size{1}; // one sample

    BytesProto _label_bytes_proto{ONE_BYTE_AS_INT};
    BytesProto _data_bytes_proto{FOUR_BYTES_AS_FLOAT};
    size_t getProtoBytes(BytesProto proto);

    size_t _minibatch{0};
   
    int *_label_buf{nullptr};
    float *_data_buf{nullptr};

    size_t _sample_iter{0};
    size_t _max_epoch{0};
    size_t _epoch{0};
    void open();
    void close();
    /*
    template <typename T>
        void read(T *buf, size_t num);
    */
    void read(size_t num);
protected:
    void restart();
public:
    DataLoader(std::string& filename, BytesProto label_proto, BytesProto data_proto, size_t epochs, size_t sample_num, const std::initializer_list<size_t>& label_shape, const std::initializer_list<size_t>& data_shape);
    bool next(int *label_batch, float *data_batch);
};
