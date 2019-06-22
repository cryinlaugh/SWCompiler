#ifndef IMPORTER_ONNXIMPORTER_H
#define IMPORTER_ONNXIMPORTER_H
#include <google/protobuf/text_format.h>
#include <unordered_map>

#include "graphIR/IRGraph.h"

namespace onnx {
class AttributeProto;
class NodeProto;
class GraphProto;
class ModelProto;
class TensorProto;
} // namespace onnx

namespace swc {

#define MAX_PROTO_SIZE 0x7FFFFFFF // 2G
/**
 *  \brief A class for importing Caffe2 Pretrained Model
 */
class ONNXImporter {
  private:
    IRGraph *graph_;
    std::unordered_map<std::string, IRNode *> name_irnode_map_;
    std::unordered_map<std::string, TensorNode *> name_tNode_map_;
    std::unordered_map<std::string, OpNode *> name_opNode_map_;
    
    void loadProto(onnx::ModelProto &net, const std::string &file);
    /// initialize network
    void loadTensors(const onnx::GraphProto &graph);
    void loadNetwork(const onnx::GraphProto &graph);
    void loadOp(const onnx::NodeProto &op);
    void loadTensor(const onnx::TensorProto &tensor);
  public:
    /** \brief build network IRGraph by loading onnx protofile
     *   \param g IRGraph to be constructed
     *   \param netProtoFile net proto
     *   \param tensorProtoFile weights proto
     *   \param udef_nodes typically input data definition
     */
    ONNXImporter(IRGraph *g, const std::string &netProtoFile,
                   std::vector<TensorNode *> &udef_nodes);
};
} // namespace swc
#endif