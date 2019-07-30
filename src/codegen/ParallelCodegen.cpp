/*************************************************************************
    > File Name: src/codegen/ParallelCodegen.cpp
    > Author: wayne
    > Mail:
    > Created Time: 二  7/30 15:12:17 2019
 ************************************************************************/

 #include "Codegen.h"
 #include "SWC.h"
 #include <fstream>

namespace swc {
namespace codegen {

bool isParallel(TensorNode * node) {
    Label *label = node->getLabel();
    Device dev = label->getDeviceLabel();
    return dev.rank == INT_MAX;
}
bool isParallel(OpNode * node) {
    // OpNode must have parent TensorNode
    auto *p0 = (TensorNode*)node->getParentNode(0);
    Device dev_p0 = p0->getLabel()->getDeviceLabel();

    // but may not have child tensor node (debug)
    if(node->childNum() > 0) {
        auto *c0 = (TensorNode*)node->getChildNode(0);
        Device dev_c0 = c0->getLabel()->getDeviceLabel();
		SWLOG_DEBUG(1) << node->name() << " isParallel = " << static_cast<int>((dev_p0.rank == INT_MAX)&&(dev_c0.rank == INT_MAX)) << "\n";
        return (dev_p0.rank == INT_MAX)&&(dev_c0.rank == INT_MAX);
    }
		SWLOG_DEBUG(1) << node->name() << " isParallel = " << static_cast<int>((dev_p0.rank == INT_MAX)) << "\n";
        return (dev_p0.rank == INT_MAX);
}

void ParallelCodegen::allocateMemAddr() {
	SWLOG_DEBUG(4) << "begin allocateMemAddr...\n";
	for (int i = 0; i < graph_->topologyNum(); i++) {
        for (int j = 0; j < graph_->getNumInTopoLevel(i); j++) {
            auto node = graph_->getNodeInTopo(i, j);
            if (node->nodeType() == TENSOR_NODE) {
                auto *tnode = (TensorNode *)node;
				Tensor *tensor = tnode->getTensor();
				if (tensors_name_map_.count(tensor))
		            continue;

				std::string buf_name = UniqueName(tnode->name());
		        size_t size = tensor->getSizeInBytes();
		        Device dev = tnode->getLabel()->getDeviceLabel();

		        SWLOG_DEBUG(1) << "allocateMemAddr " << tnode->name() << " " << size
		                       << " on dev(" << dev.rank << ", "
		                       << static_cast<int>(dev.type) << ", "
		                       << dev.id << ")."
		                       << "\n";

				auto *allocator = dev_allocator_map_.at(dev);
			    if (!allocator) {
			        SWLOG_ERROR << "allocator" << static_cast<int>(dev.type) << " "
			                    << dev.id << " not found\n";
			    }
			    uint64_t addr = allocator->allocate(tensor, size);
			    std::string base = allocator->getBasePtrName();

			    tensors_name_map_[tensor] = buf_name;
			    tensors_offset_map_[tensor] = std::make_pair(base, addr);

				if(dev.rank == 0) {
					_master_tensors.push_back(tnode);
				} else {
					_parallel_tensors.push_back(tnode);
				}
			}
		}
	}

	SWLOG_DEBUG(4) << "end allocateMemAddr...\n";
}

void ParallelCodegen::emitVarDeclarations() {
	SWLOG_DEBUG(4) << "begin emitVarDeclarations...\n";

	for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (size == 0)
            continue;
        writer_ << "char *" << base << ";\n";
    }

	if(p_mem_alllocator_->getMemAllocated()) {
        std::string base = p_mem_alllocator_->getBasePtrName();
        writer_ << "char *" << base << ";\n";
    }

	for (auto it : tensors_name_map_) {
        auto *tensor = it.first;
        std::string dtype = getTypeString(tensor);
        writer_ << dtype << " *" << it.second << ";\n";
    }

	writer_ << "\n";
    SWLOG_DEBUG(4) << "end emitVarDeclarations...\n";
}

void ParallelCodegen::emitMemAllocations() {
	SWLOG_DEBUG(4) << "begin emitMemAllocations...\n";

	writer_ << "if(rank == 0) {\n";
	writer_.indentInc();
	for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (dev.rank != 0 || size == 0)
            continue;

        emitMemAllocation(base, size, dev);
    }

	writer_.indentDec();
	writer_ << "} // if rank\n";

	writer_ << "if(rank != 0) {\n";
	writer_.indentInc();

	if(p_mem_alllocator_->getMemAllocated()) {

        auto dev = p_mem_alllocator_->getDevice();
        std::string base = p_mem_alllocator_->getBasePtrName();
        uint64_t size = p_mem_alllocator_->getMemAllocated();

		emitMemAllocation(base, size, dev);

    }
	// for (auto m : mem_allocators_) {
    //     MemoryAllocator *allocator = m.get();
    //     auto dev = allocator->getDevice();
    //     std::string base = allocator->getBasePtrName();
    //     uint64_t size = allocator->getMemAllocated();
    //     if (dev.rank == 0 || size == 0)
    //         continue;
	//
    //     emitMemAllocation(base, size, dev);
    // }
	writer_.indentDec();
	writer_ << "} // if rank\n";

	writer_ << "\n";
	SWLOG_DEBUG(4) << "end emitMemAllocations...\n";
}

void ParallelCodegen::emitTensorAddresses() {
	SWLOG_DEBUG(4) << "begin emitTensorAddresses...\n";
	writer_ << "if(rank == 0) {\n";
	writer_.indentInc();

	for(auto *tnode : _master_tensors) {
		auto *tensor = tnode->getTensor();
		std::string dtype = getTypeString(tensor);

        std::string name = tensors_name_map_[tensor];
        std::string base;
        uint64_t offset;
        std::tie(base, offset) = tensors_offset_map_[tensor];
        writer_ << name << " = reinterpret_cast<" << dtype << "*>(" << base
                << " + " << offset << ");\n";
	}

	writer_.indentDec();
	writer_ << "} // if rank\n";

	writer_ << "if(rank != 0) {\n";
	writer_.indentInc();

	for(auto *tnode : _parallel_tensors) {
		auto *tensor = tnode->getTensor();
		std::string dtype = getTypeString(tensor);

        std::string name = tensors_name_map_[tensor];
        std::string base;
        uint64_t offset;
        std::tie(base, offset) = tensors_offset_map_[tensor];
        writer_ << name << " = reinterpret_cast<" << dtype << "*>(" << base
                << " + " << offset << ");\n";
	}


	writer_.indentDec();
	writer_ << "} // if rank\n";

	writer_ << "\n";
	SWLOG_DEBUG(4) << "end emitTensorAddresses...\n";
}

void ParallelCodegen::emitTensorInitializations() {
	SWLOG_DEBUG(4) << "begin emitTensorInitializations...\n";
	writer_ << "if(rank == 0) {\n";
	writer_.indentInc();

	for(auto *tnode : _master_tensors) {
		emitTensorInitialization(tnode);
	}
	writer_ << "\n";
	for(auto *tnode : _parallel_tensors) {
		auto *tensor = tnode->getTensor();
		if(tensor->getTensorInitType() == TensorInitType::PARENTOP) {
			auto *parent = (OpNode*) tnode->getParentNode(0);
			if (auto *scatter = dynamic_cast<ScatterOp *>(parent->getOp())) {
		    	writer_ << "// master to worker send statements\n";
				masterWorkerDispatcher(parent, 0);
			}
		}

	}

	writer_.indentDec();
	writer_ << "} // if rank == 0\n";

	writer_ << "\n";

	writer_ << "if(rank != 0) {\n";
	writer_.indentInc();

	for(auto *tnode : _parallel_tensors) {
		auto *tensor = tnode->getTensor();
		if(tensor->getTensorInitType() == TensorInitType::PARENTOP) {
			auto *parent = (OpNode*) tnode->getParentNode(0);
			if (auto *scatter = dynamic_cast<ScatterOp *>(parent->getOp())) {
		    	writer_ << "// worker recv from master statements\n";
				if(parent->runable())
					masterWorkerDispatcher(parent, 1);
				continue;
			}
		}
		emitTensorInitialization(tnode);
	}

	writer_.indentDec();
	writer_ << "} // if rank != 0\n";

	writer_ << "\n";
	SWLOG_DEBUG(4) << "end emitTensorInitializations...\n";
}


void ParallelCodegen::masterWorkerDispatcher(OpNode *op, int side/*master:0, worker:1*/) {
	if (auto *scatter = dynamic_cast<ScatterOp *>(op->getOp())) {
		auto *from = ((TensorNode *)op->getParentNode(0));
		auto *from_tensor = from->getTensor();
		// Device from_dev = from->getLabel()->getDeviceLabel();
		auto *to = ((TensorNode *)op->getChildNode(0));
		auto *to_tensor = to->getTensor();

		std::string fname = tensors_name_map_[from_tensor];
		std::string tname = tensors_name_map_[to_tensor];

		int axis = scatter->getAxis();
		int degree = scatter->getDegree();
		size_t size = to_tensor->getSizeInBytes();
		size_t offset = (axis == -1) ? 0 : to_tensor->size();

		int tag = getMPISendRecvTag(to_tensor);

		if(side == 0) {
			writer_ << "for(int r=1; r<=" << degree <<"; r++) {\n";
			writer_.indentInc();
			writer_ << "MPI_Send(" << fname << "+(r-1)*" << offset << ", " << size
				   << ", "
				   << "MPI_CHAR, r, " << tag
				   << ",  MPI_COMM_WORLD);\n";
			writer_.indentDec();
			writer_ << "} //for \n";
		} else {
			writer_ << "MPI_Recv(" << tname << ", " << size
				   << ", "
				   << "MPI_CHAR, 0, "<< tag
				   << ",  MPI_COMM_WORLD, &status);\n";
		}
	} else if (auto gather = dynamic_cast<GatherOp *>(op->getOp())) {
	   auto *from = ((TensorNode *)op->getParentNode(0));
	   auto *from_tensor = from->getTensor();
	   auto *to = ((TensorNode *)op->getChildNode(0));
	   auto *to_tensor = to->getTensor();
	   // Device to_dev = to->getLabel()->getDeviceLabel();

	   std::string fname = tensors_name_map_[from_tensor];
	   std::string tname = tensors_name_map_[to_tensor];
	   SWLOG_DEBUG(2) << "gather\n";

	   // TODO, non-continuous
	   int axis = gather->getAxis();
	   int degree = gather->getDegree();
	   size_t size = from_tensor->getSizeInBytes();
	   size_t offset = (axis == -1) ? 0 : from_tensor->size();

	   SWLOG_DEBUG(2) << "GatherOp axis=" <<axis << " degree=" << degree << "\n";

	   int tag = getMPISendRecvTag(from_tensor);

	   if(side == 0) {
		   writer_ << "for(int r=1; r<=" << degree <<"; r++) {\n";
		   writer_.indentInc();
		   writer_ << "MPI_Recv(" << tname << "+(r-1)*" << offset <<", " << size
				   << ", "
				   << "MPI_CHAR, r, "<< tag
				   << ",  MPI_COMM_WORLD, &status);\n";
		   writer_.indentDec();
		   writer_ << "} //for \n";
	   } else {
		   writer_ << "MPI_Send(" << fname << ", " << size
				   << ", "
				   << "MPI_CHAR, 0, " << tag
				   << ",  MPI_COMM_WORLD);\n";
	   }
   }

}


void ParallelCodegen::emitTensorInitialization(TensorNode *tnode) {
	auto *tensor = tnode->getTensor();

	std::string dtype = getTypeString(tensor);
	std::string name = tensors_name_map_[tensor];
	uint64_t size = tensor->size();
	std::string base;
	uint64_t offset;
	std::tie(base, offset) = tensors_offset_map_[tensor];

	TensorInitInfo info = tensor->getTensorInitInfo();
	switch (tensor->getTensorInitType()) {
	case TensorInitType::NONE:
		break;
	case TensorInitType::XAVIER: {
		// TODO
		writer_ << "initTensorXavier(" << name << ", " << size << ", "
				<< info.getFilterSize() << ");\n";
		break;
	}
	case TensorInitType::CONSTANT: {
		writer_ << "initTensorConstant(" << name << ", " << size << ", "
				<< info.getConstant() << ");\n";
		break;
	}
	case TensorInitType::ZERO: {
		writer_ << "initTensorZero(" << name << ", " << size << ");\n";
		break;
	}
	case TensorInitType::FILE: {
		writer_ << "load(" << name << ", " << size << ", "
				<< info.getOffset() << ", "
				<< "\"" << info.getFilePath() << "\");\n";
		break;
	}
	case TensorInitType::PARENTOP: {
		// auto *op = (OpNode *)tnode->getParentNode(0);
		// dispatchOpNode(op);
		break;
	}
	default:
		SWLOG_DEBUG(1) << name << " TensorInitType= NONE\n";
		break;

	} // switch
}

std::vector<OpNode*> ParallelCodegen::schedule() {
	std::vector<OpNode*> scheduled_nodes;
	for(int i=0; i<graph_->topologyNum(); i++) {
		for(int j=0; j<graph_->getNumInTopoLevel(i); j++) {
			auto *node = graph_->getNodeInTopo(i, j);
			// if(node->nodeType() == OP_NODE) {
			if (auto *opnode = dynamic_cast<OpNode *>(node)) {
				scheduled_nodes.push_back(opnode);
			}
		}
	}
	return scheduled_nodes;
}

void ParallelCodegen::emitFuncCalls() {
	SWLOG_DEBUG(4) << "begin emitFuncCalls...\n";

	auto scheduled = schedule();

	writer_ << "if(rank == 0) {\n";
	writer_.indentInc();

	SWLOG_DEBUG(4) << ">>>> rank 0 dispatching\n";
	for(auto *node : scheduled) {
		if (dynamic_cast<ScatterOp *>(node->getOp()) ||
				dynamic_cast<GatherOp *>(node->getOp())) {
			writer_ << "// " << node->name() << "\n";
			dispatchOpNode(node, 0);
			// masterWorkerDispatcher(node, 0);
			continue;
		}
		if(isParallel(node))
			continue;

		writer_ << "// " << node->name() << "\n";
		dispatchOpNode(node, 0);
	}

	writer_.indentDec();
	writer_ << "} // if rank == 0\n";

	writer_ << "\n";

	SWLOG_DEBUG(4) << ">>>> rank parallel dispatching\n";
	writer_ << "if(rank != 0) {\n";
	writer_.indentInc();

	for(auto *node : scheduled) {
		if (dynamic_cast<ScatterOp *>(node->getOp()) ||
				dynamic_cast<GatherOp *>(node->getOp())) {
			writer_ << "// " << node->name() << "\n";
			dispatchOpNode(node, 1);
			// masterWorkerDispatcher(node, 0);
			continue;
		}

		if(isParallel(node)) {
			writer_ << "// " << node->name() << "\n";
			dispatchOpNode(node, 1);
		}

	}

	writer_.indentDec();
	writer_ << "} // if rank != 0\n";

	writer_ << "\n";

	SWLOG_DEBUG(4) << "end emitFuncCalls...\n";
}

void ParallelCodegen::dispatchOpNode(OpNode *op, int side/*0:master, ~0: worker*/) {
    if (!op->runable())
        return;

	SWLOG_DEBUG(4) << "dispatchOpNode " << op->name() << " for rank " << side << "\n";

    Label *label = op->getLabel();
    Device dev = label->getDeviceLabel();
    if (dynamic_cast<ScatterOp *>(op->getOp()) ||
			dynamic_cast<GatherOp *>(op->getOp())) {
		masterWorkerDispatcher(op, side);
    }
	// else if (auto gather = dynamic_cast<TransformOp *>(op->getOp())) {
    // }
	else {
        switch (dev.type) {
        case DeviceType::CPU:
            emitFuncCall(op);
            break;
        case DeviceType::GPU:
            emitFuncCallCUDA(op);
            break;
        default:
            SWLOG_ERROR << "unknown device type in dispatchOpNode\n";
        }
    }
}

void ParallelCodegen::emitMemFree(std::string name, Device dev) {
	switch (dev.type) {
	case DeviceType::CPU:
		writer_ << "free(" << name << ");\n";
		break;
	case DeviceType::GPU:
		writer_ << "\n";
		writer_ << "cudaSetDevice(" << dev.id << ");\n";
		writer_ << "cudaFree(" << name << ");\n";
		break;
	default:
		SWLOG_ERROR << "Unknown DeviceType\n";
		break;
	}
}

void ParallelCodegen::emitMemFree() {
	SWLOG_DEBUG(4) << "begin emitMemFree...\n";

	writer_ << "if(rank == 0) {\n";
	writer_.indentInc();
	for (auto m : mem_allocators_) {
        MemoryAllocator *allocator = m.get();
        auto dev = allocator->getDevice();
        std::string base = allocator->getBasePtrName();
        uint64_t size = allocator->getMemAllocated();
        if (dev.rank != 0 || size == 0)
            continue;
		emitMemFree(base, dev);

    }

	writer_.indentDec();
	writer_ << "} // if rank\n";

	writer_ << "if(rank != 0) {\n";
	writer_.indentInc();

	if(p_mem_alllocator_->getMemAllocated()) {

        auto dev = p_mem_alllocator_->getDevice();
        std::string base = p_mem_alllocator_->getBasePtrName();
        // uint64_t size = p_mem_alllocator_->getMemAllocated();

		emitMemFree(base, dev);

    }
	// for (auto m : mem_allocators_) {
    //     MemoryAllocator *allocator = m.get();
    //     auto dev = allocator->getDevice();
    //     std::string base = allocator->getBasePtrName();
    //     uint64_t size = allocator->getMemAllocated();
    //     if (dev.rank == 0 || size == 0)
    //         continue;
	//
    //     emitMemAllocation(base, size, dev);
    // }
	writer_.indentDec();
	writer_ << "} // if rank\n";

	writer_ << "\n";
	SWLOG_DEBUG(4) << "end emitMemFree...\n";
}

} // namespace codegen
} // namespace swc
