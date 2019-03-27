/*************************************************************************
	> File Name: MemoryAllocator.h
	> Author: wayne
	> Mail:  
	> Created Time: 四  3/14 16:03:50 2019
 ************************************************************************/
#ifndef _MEMORY_ALLOCATOR_H_
#define _MEMORY_ALLOCATOR_H_
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <string>
#include <assert.h>
#include "common.h"

// TODO: implement aligned allocate and deallocate
// naive impl, simply memAccumulator
class MemoryAllocator{
    Device dev_;
    std::string name_;
    uint64_t capacity_;
    uint64_t allocated_{0};

    std::string baseptr_name_;

    std::unordered_map<const void*, uint64_t > mem_addr_map_;
    std::unordered_map<uint64_t , const void*> addr_mem_map_; 
    
    void clear();
public:
    MemoryAllocator(uint64_t cap) : capacity_(cap) {}
    MemoryAllocator(Device& dev, std::string name, uint64_t cap) 
        : dev_(dev), name_(name), capacity_(cap) {}
    ~MemoryAllocator(){ clear(); }
    uint64_t allocate(const void* mem, uint64_t size);
    uint64_t getMemAllocated() { return allocated_; }
    void setBasePtrName(std::string name) { baseptr_name_ = name; }
    std::string getBasePtrName() {return baseptr_name_; }
    Device& getDevice() { return dev_; }
};  
#endif
