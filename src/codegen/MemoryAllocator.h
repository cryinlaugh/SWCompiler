/*************************************************************************
    > File Name: MemoryAllocator.h
    > Author: wayne
    > Mail:
    > Created Time: 四  3/14 16:03:50 2019
 ************************************************************************/
#ifndef _MEMORY_ALLOCATOR_H_
#define _MEMORY_ALLOCATOR_H_
#include "common.h"
#include <assert.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

/**
 * \brief aggregated memory allocator for continuous tensor memory
 *  naive implementation, simply memAccumulator
 *  TODO: implement aligned allocate and deallocate
 */
class MemoryAllocator {
    Device dev_; ///< abstract of Device
    std::string name_;
    uint64_t capacity_;     ///< max memory (bytes)
    uint64_t allocated_{0}; ///< allocated memory (bytes)

    std::string
        baseptr_name_; ///< baseptr_name_ memory allocation code emitting

    std::unordered_map<const void *, uint64_t>
        mem_addr_map_; ///< Tensor* -> offset map
    std::unordered_map<uint64_t, const void *>
        addr_mem_map_; ///< offset -> Tensor* map

    void clear();

  public:
    MemoryAllocator(uint64_t cap) : capacity_(cap) {}
    MemoryAllocator(Device &dev, std::string name, uint64_t cap)
        : dev_(dev), name_(name), capacity_(cap) {}
    ~MemoryAllocator() { clear(); }
    /// allocate start addr(offset) for mem
    uint64_t allocate(const void *mem, uint64_t size);
    uint64_t getMemAllocated() { return allocated_; }
    void setBasePtrName(std::string name) { baseptr_name_ = name; }
    std::string getBasePtrName() { return baseptr_name_; }
    Device &getDevice() { return dev_; }
    uint64_t getCapacity() { return capacity_; }
};
#endif
