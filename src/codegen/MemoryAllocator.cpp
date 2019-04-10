#include "MemoryAllocator.h"

void MemoryAllocator::clear() {
    allocated_ = 0;
    mem_addr_map_.clear();
    addr_mem_map_.clear();
}
uint64_t MemoryAllocator::allocate(const void *mem, uint64_t size) {
    uint64_t addr = allocated_;
    allocated_ += size;
    mem_addr_map_[mem] = addr;
    addr_mem_map_[addr] = mem;

    return addr;
}