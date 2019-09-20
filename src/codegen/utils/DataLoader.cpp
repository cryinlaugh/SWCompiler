/*************************************************************************
    > File Name: DataLoader.cpp
    > Author: wayne
    > Mail:
    > Created Time: 四  5/16 20:55:42 2019
 ************************************************************************/
#include "DataLoader.h"
#include <cassert>
#include <iostream>

DataLoader::DataLoader(std::string &filename, BytesProto label_proto,
                       BytesProto data_proto, size_t epochs, size_t sample_num,
                       const std::initializer_list<size_t> &label_shape,
                       const std::initializer_list<size_t> &data_shape) {
    assert(label_shape.size() && data_shape.size());
    for (auto i : label_shape)
        _label_batch_shape.push_back(i);
    for (auto i : data_shape)
        _data_batch_shape.push_back(i);
    assert(_label_batch_shape[0] == _data_batch_shape[0]);

    _label_bytes_proto = label_proto;
    _data_bytes_proto = data_proto;

    _datafile = filename;

    _sample_num = sample_num;
    _minibatch = _data_batch_shape[0];

    _label_bytes = getProtoBytes(_label_bytes_proto);
    for (size_t i = 1; i < _data_batch_shape.size(); i++)
        _data_size *= _data_batch_shape[i];
    _data_bytes = _data_size * getProtoBytes(_data_bytes_proto);

    _max_epoch = epochs;
}

size_t DataLoader::getProtoBytes(BytesProto proto) {
    switch (proto) {
    case ONE_BYTE_AS_INT:
        return 1;
    case FOUR_BYTES_AS_FLOAT:
        return 4;
    default:
        return 1;
    }
}

void DataLoader::check_init() {
    if(init_flag)
        return;

    _stream.open(_datafile, std::ios::binary);
    if (!_stream.is_open()) {
        std::cout << "Error loading " << _datafile << "\n";
        std::exit(EXIT_FAILURE);
    }
    init_flag = true;
    std::cout << "DataLoader Summary: \n"
              << "<" << _label_bytes << " x label><" << _data_bytes << " x "
              << "data>"
              << "\n"
              << "_minibatch: " << _minibatch << "\n"
              << "_label_bytes: " << _label_bytes << "\n"
              << "_data_size: " << _data_size << "\n"
              << "_data_bytes: " << _data_bytes << "\n";

}

void DataLoader::close() {
    if (_stream.is_open())
        _stream.close();
}

void DataLoader::restart() {
    _epoch++;
    _sample_iter = 0;
    _stream.seekg(0, _stream.beg);
}

/*
/// 1 byte to int
template <>
void DataLoader::read<int>(int *buf, size_t num) {
    char c;
    for(size_t i=0; i<num; i++) {
        _stream.get(c);
       buf[i] = static_cast<int>(c);
    }
}

/// 4 bytes (little endian 78 56 34 12 -> 0x12345678) to float
template <>
void DataLoader::read<float>(float *buf, size_t num) {
    _stream.read(reinterpret_cast<char*>(buf), num * _data_bytes);
}
*/
void DataLoader::read(size_t num) {
    char c;
    for (size_t i = 0; i < num; i++) {
        _stream.get(c);
        _label_buf[i] = static_cast<int>(c);
        switch (_data_bytes_proto) {
        case FOUR_BYTES_AS_FLOAT:
            _stream.read(reinterpret_cast<char *>(_data_buf + i * _data_size),
                         _data_bytes);
            break;
        case ONE_BYTE_AS_INT:
            for (size_t j = 0; j < _data_size; j++) {
                _stream.get(c);
                _data_buf[i * _data_size + j] =
                    static_cast<float>(static_cast<int>(c));
            }
        default:
            break;
        }
    }
}

bool DataLoader::next(int *label_batch, float *data_batch) {
    check_init();

    // std::cout << "loader want to fill " << label_batch << " and " << data_batch;
    if (_epoch == _max_epoch) {
        close();
        return false;
    }

    size_t batch_res = _minibatch;
    _label_buf = label_batch;
    _data_buf = data_batch;

    while (batch_res > 0) {
        size_t sample_res = _sample_num - _sample_iter;
        if (sample_res == 0) {
            restart();
            if (_epoch == _max_epoch) {
                close();
                return false;
            }
            continue;
        }
        size_t actual_read_cnt = std::min(batch_res, sample_res);

        read(actual_read_cnt);
        /*
        read(label_buf, actual_read_cnt * _label_bytes);
        read(data_buf, actual_read_cnt * _data_bytes);
        */

        _label_buf += actual_read_cnt;
        _data_buf +=
            actual_read_cnt * _data_size; // cnt * element_num_per_sample
        _sample_iter += actual_read_cnt;
        batch_res -= actual_read_cnt;
    }

    return true;
}

void DataLoader::shift(size_t batch) {
    size_t n = _minibatch * batch;
    _epoch = n / _sample_num;
    _sample_iter = n % _sample_num;

    size_t one_sample_bytes = _label_bytes + _data_bytes;
    _stream.seekg(one_sample_bytes * _sample_iter, _stream.beg);
}
