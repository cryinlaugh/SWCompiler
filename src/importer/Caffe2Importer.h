/*************************************************************************
	> File Name: Caffe2Loader.h
	> Author: wayne
	> Mail:  
	> Created Time: 四  3/28 10:12:10 2019
 ************************************************************************/
#ifndef IMPORTER_CAFFE2IMPORTER_H
#define IMPORTER_CAFFE2IMPORTER_H
#include <google/protobuf/text_format.h>
#include <unordered_map>

#include "graphIR/IRGraph.h"

namespace caffe2{
class NetDef;
class OperatorDef;
}
/*
namespace swc{
    template <typename T> class IRGraph;
    template <> class IRGraph<float>;
}
*/

namespace swc{

#define MAX_PROTO_SIZE 0x7FFFFFFF //2G
class Caffe2Importer{
private:
    IRGraph *graph_;
    std::unordered_map<std::string, IRNode*> name_irnode_map_;
    std::unordered_map<std::string, TensorNode*> name_tNode_map_;
    std::unordered_map<std::string, OpNode*> name_opNode_map_;
    void loadProto(caffe2::NetDef &net, const std::string &file);
    void loadNetwork(caffe2::NetDef &net);
    void loadTensors(caffe2::NetDef &tensors);
    void loadOp(const caffe2::OperatorDef &op);
    void loadTensor(const caffe2::OperatorDef &op);
public:
    Caffe2Importer(IRGraph *g, const std::string &netProtoFile,
                 const std::string &tensorProtoFile, std::vector<TensorNode*> &udef_nodes);
};
}
#endif
