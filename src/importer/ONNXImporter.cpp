#include "ONNXImporter.h"
#include "onnx-ml.pb.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <iostream>
#include <string>
#include <vector>

#include "SWDSL.h"
#include "common.h"
#include "graphIR/IRNode.h"
#include "graphIR/OpNode.h"
#include "graphIR/TensorNode.h"
#include "op/Op.h"
#include "op/dlOp/dlOp.h"
#include "tensor/tensor.h"

namespace swc {

using ArgumentMap = std::unordered_map<std::string, const onnx::AttributeProto *>;

std::vector<size_t> getPads(const ArgumentMap &args) {
    if (args.count("pads")) {
        std::vector<size_t> pads;
        for (auto i : args.at("pads")->ints())
            pads.push_back(i);
        return pads;
    }
    return {0, 0, 0, 0};
}

// kernel for Conv and Pooling
std::vector<size_t> getKernels(const ArgumentMap &args) {
    if (args.count("kernel_shape")) {
        auto *attri = args.at("kernel_shape");
        if (attri->has_i()) {
            int value = attri->i();
            std::vector<size_t> kernels(2, value);
            return kernels;
        }
        else if (attri->ints_size() > 0) {
            // ints is deprecated now, but some models still have this.
            std::vector<size_t> kernels;
            for (auto i : attri->ints())
                kernels.push_back(i);
            return kernels;
        }
    }
    return {0, 0};
}
std::vector<size_t> getStrides(const ArgumentMap &args) {
    if (args.count("strides")) {
        auto *attri = args.at("strides");
        if (attri->ints_size() > 0) {
            // ints is deprecated now, but some models still have this.
            std::vector<size_t> strides;
            for (auto i : attri->ints())
                strides.push_back(i);
            return strides;
        }
        // int value = args.at("strides")->ints(0);
        // std::vector<size_t> strides(2, value);
        // return strides;
    }
    return {1, 1};
}

std::vector<size_t> inferConvOutDims(size_t ih, size_t iw,
                                            std::vector<size_t> &kernels,
                                            std::vector<size_t> &strides,
                                            std::vector<size_t> &pads) {
    assert(kernels.size() == 2);
    assert(strides.size() == 2);
    assert(pads.size() == 4);

    size_t oh = ((ih + pads[0] + pads[2] - kernels[0]) / strides[0] + 1);
    size_t ow = ((iw + pads[1] + pads[3] - kernels[1]) / strides[1] + 1);
    return {oh, ow};
}

ONNXImporter::ONNXImporter(IRGraph *g, const std::string &netProtoFile,
                               std::vector<TensorNode *> &udef_nodes) {
    graph_ = g;

    for (auto tnode : udef_nodes) {
        graph_->pushTensorNode(tnode);
        std::string name = tnode->name();

        name_tNode_map_[name] = tnode;
    }

    size_t err = system("mkdir /tmp/SW");
    if(err == 0) {
        SWLOG_INFO << "Create directory /tmp/SW/\n";
    } else {
        SWLOG_INFO << "Directory /tmp/SW/ already exists, go on\n";
    }

    onnx::ModelProto net;
    onnx::GraphProto graph;
    loadProto(net, netProtoFile);
    graph = net.graph();
    loadTensors(graph);
    loadNetwork(graph);
}

void ONNXImporter::loadProto(onnx::ModelProto &net,
                               const std::string &filename) {
    std::ifstream ff(filename, std::ios::in | std::ios::binary);
    assert(ff && "Can't find the model or network files.");
    bool parseNet = false;
    google::protobuf::io::IstreamInputStream fileStream(&ff);
    google::protobuf::io::CodedInputStream codedstr(&fileStream);
    // Don't warn about large file sizes.
    codedstr.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
    parseNet = net.ParseFromCodedStream(&codedstr);

    assert(parseNet && "Failed to parse the network descriptor.");
}

void ONNXImporter::loadTensors(const onnx::GraphProto &graph) {
    for(const auto &tensor : graph.initializer()) {
        loadTensor(tensor);
    }
}

void ONNXImporter::loadTensor(const onnx::TensorProto &tensor) {
    SWLOG_DEBUG(4) << "load Tensor " << tensor.name() << "\n"; 
    
    auto *T = new Tensor();
    auto *shape = new std::vector<size_t>();
    size_t size = 1;
    for(auto d : tensor.dims()) {
        shape->push_back(d);
        size *= d;
    }
    T->reset(new TensorShape(shape));

    // write tensor init value to file
    std::ostringstream address;
    address << (void const *)T;
    std::string path = "/tmp/SW/" + address.str();
    T->setTensorInit(TensorInitType::FILE, path);
    std::ofstream fout(path, std::ios::out | std::ios::binary);

    if(tensor.data_type() == onnx::TensorProto::FLOAT) {
        std::vector<float> tensorValue;
        if(tensor.float_data_size() > 0) {
            for (auto f : tensor.float_data()) {
                float v = f;
                tensorValue.push_back(v);
            }
        } else if (tensor.has_raw_data()) {
            tensorValue.resize(size);
            std::istringstream inStream(tensor.raw_data(), std::stringstream::binary);
            inStream.read((char *)&tensorValue[0], size * sizeof(float));
        }

        fout.write((char *)&tensorValue[0], tensorValue.size() * sizeof(float));
        SWLOG_DEBUG(4) << "write to " << path << " float*" << tensorValue.size() << "\n";  
        tensorValue.clear();

    } else if(tensor.data_type() == onnx::TensorProto::INT64) {
        // std::vector<int64_t> tensorValue;
        // for (auto i : tensor.int64_data()) {
        //     float v = i;
        //     tensorValue.push_back(v);
        // }
        // fout.write((char *)&tensorValue[0], tensorValue.size() * sizeof(int64_t));
        // tensorValue.clear();
    } else {

    }
    fout.close();

    std::string name = tensor.name();
    if(!name_tNode_map_.count(name)) {
        auto *tnode = new TensorNode(name, T);
        graph_->pushTensorNode(tnode);
        name_tNode_map_[name] = tnode;
    }
}

void ONNXImporter::loadNetwork(const onnx::GraphProto &graph) {
    for (const auto &op : graph.node()) {
        loadOp(op);
    }
}

void ONNXImporter::loadOp(const onnx::NodeProto &op) {
    SWLOG_DEBUG(4) << "load Op " << op.op_type() << " " << op.output(0) << "\n";
    std::string opType = op.op_type();

    // name() is often null; output(0) may be repeated.
    // std::string opName = opType;
    // std::transform(opName.begin(), opName.end(), opName.begin(), ::tolower);
    std::string opName = op.output(0);

    std::cout << opName << " " << op.output(0) << std::endl
              << "\ttype  : " << op.op_type() << std::endl
              << "\tinput : " << op.input_size()<< " ";
    for(auto &input : op.input()) {
        std::cout << input << "/";
    }          
    std::cout << std::endl;
    std::cout << "\toutput: " << op.output_size() << " ";
    for(auto &output : op.output()) {
        std::cout << output << "/";
    } 
    std::cout << std::endl;


    std::unordered_map<std::string, const onnx::AttributeProto *> args;
    std::cout << "\tattribute: ";
    for(auto &arg : op.attribute()) {
        assert(arg.has_name() && "Attribute without name!");
        std::cout << arg.name() << "/";
        args[arg.name()] = &arg;
    }
    std::cout << std::endl;

    OpNode *opNode;
    if(opType == "Conv") {
        // assert(op.input_size() == 3 && "conv bias is needed!!");
        auto data = name_tNode_map_[op.input(0)];
        auto weight = name_tNode_map_[op.input(1)];

        std::vector<size_t> inDims = data->getDims();

        TensorNode *bias;
        if (op.input_size() == 3) {
            bias = name_tNode_map_[op.input(2)];
        } else {
            std::string nm = opName + "_bias";
            bias = new TensorNode(nm, {inDims[3]});
            graph_->pushTensorNode(bias);
        }

        std::vector<size_t> kernels = getKernels(args);
        std::vector<size_t> strides = getStrides(args);
        std::vector<size_t> pads = getPads(args);

        

        std::string trans_op_name = "op_" + weight->name() + "_T";
        auto trans = new OpNode(trans_op_name, new TransposeOp(NCHW2NHWC));
        LINKUPPER(trans, weight);

        Tensor *wt =
            new Tensor(weight->getTensor()->getShuffledTensorShape(NCHW2NHWC));
        std::string trans_name = weight->name() + "_T";
        auto w_trans = new TensorNode(trans_name, wt, trans);

        auto *convOp = new Conv2dOp(kernels, strides, pads);
        opNode = new OpNode(opName, convOp);
        opNode->exlinkUpperNode(data, w_trans, bias);

        std::vector<size_t> ohw =
            inferConvOutDims(inDims[1], inDims[2], kernels, strides, pads);

        graph_->pushOpNode(trans);
        graph_->pushTensorNode(w_trans);

        size_t n = inDims[0]; // from data
        size_t c = w_trans->getDims()[0];

        std::string res_name = op.output(0);
        auto out_tnode =
            new TensorNode(res_name, {n, ohw[0], ohw[1], c}, opNode);
        name_tNode_map_[res_name] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }

    else if (opType == "MaxPool") {
        auto in = name_tNode_map_[op.input(0)];

        std::vector<size_t> kernels = getKernels(args);
        std::vector<size_t> strides = getStrides(args);
        std::vector<size_t> pads = getPads(args);

        auto *poolOp = new MaxPoolOp(kernels, strides, pads);
        opNode = new OpNode(opName, poolOp);
        LINKUPPER(opNode, in);

        std::vector<size_t> inDims = in->getDims();
        size_t n = inDims[0];
        size_t c = inDims[3];
        std::vector<size_t> ohw =
            inferConvOutDims(inDims[1], inDims[2], kernels, strides, pads);

        std::string res_name = op.output(0);
        auto *out_tnode =
            new TensorNode(res_name, {n, ohw[0], ohw[1], c}, opNode);
        name_tNode_map_[res_name] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }

    else if (opType == "Gemm") {
        /**
         * e.g.
         * https://github.com/onnx/onnx/blob/master/docs/Operators.md#gemm
            type  : Gemm
            input : 3 OC2_DUMMY_0/fc3_w_0/fc3_b_0/
            output: 1 fc3_1/
            attribute: transB/broadcast/
         */
        
        opNode = new OpNode(opName, new MatrixMatrixFCOp());

        assert(op.input_size() == 3 && "Gemm bias is needed!!");
        auto in = name_tNode_map_[op.input(0)];
        auto weight = name_tNode_map_[op.input(1)];
        auto bias = name_tNode_map_[op.input(2)];

        auto alpha = args.count("alpha") ? args.at("alpha")->f() : 1.0; (void)alpha;
        auto beta = args.count("beta") ? args.at("beta")->f() : 1.0;   (void)beta;
        auto transA = args.count("transA") ? args.at("transA")->i() : 0;   (void)transA;
        auto transB = args.count("transB") ? args.at("transB")->i() : 0;   (void)transB;

        assert(transB && "suppose transB==1 in ONNX Gemm");

        std::string trans_op_name = "op_" + weight->name() + "_T";
        auto trans = new OpNode(trans_op_name, new TransposeOp({1, 0}));
        LINKUPPER(trans, weight);

        Tensor *wt =
            new Tensor(weight->getTensor()->getShuffledTensorShape({1, 0}));
        std::string trans_name = weight->name() + "_T";
        auto w_trans = new TensorNode(trans_name, wt, trans);

        // since we always have reshape node before 4d tensor and fc
        opNode->exlinkUpperNode(in, w_trans, bias);
        graph_->pushOpNode(trans);
        graph_->pushTensorNode(w_trans);


        std::vector<size_t> inDims = in->getDims();
        std::vector<size_t> weightDims = w_trans->getDims();
        size_t flatdim = 1;
        for (size_t i = 1; i < inDims.size(); i++)
            flatdim *= inDims[i];

        SWLOG_DEBUG(2) << "FC, flatdim=" << flatdim
                       << " weightDims[0]=" << weightDims[0] << "\n";
        assert((flatdim == weightDims[0]) &&
               "input flattenedDim not equal with weight\n");

        size_t n = inDims[0]; // from data
        size_t sec = weightDims[1];

        std::string res_name = op.output(0);
        auto *out_tnode = new TensorNode(res_name, {n, sec}, opNode);
        name_tNode_map_[res_name] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }

    else if (opType == "Relu") {
        opNode = new OpNode(opName, new ReluOp());

        std::string iname = op.input(0);
        auto *in = name_tNode_map_[iname];
        LINKUPPER(opNode, in);

        std::string res_name = op.output(0);
        TensorNode *out_tnode;
        if (name_tNode_map_.count(res_name)) {
            auto *tensor = name_tNode_map_[res_name]->getTensor();
            out_tnode = new TensorNode(res_name, tensor, opNode);
        } else {
            auto *tshape = in->getTensor()->getTensorShape();
            out_tnode = new TensorNode(res_name, new Tensor(tshape), opNode);
        }
        // name mapping to newest operator's  res TensorNode
        name_tNode_map_[res_name] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }

    else if (opType == "Reshape") {
        opNode = new OpNode(opName);
        assert(op.input_size() == 2 && "Reshaped is needed!!");
        auto in = name_tNode_map_[op.input(0)];
        auto shape = name_tNode_map_[op.input(1)];
        // ONNX store 4D tensor as NCHW, 
        // but TensorProto lacks field to indicate layout
        // when reshape 4D to 2D, we implicitly do this:
        // NHWC -> NCHW -> N*(CHW)

        // Experiment show that this is not reliable
        // e.g. trans 8*4*4*50 to 8*800
        // you may expect shape dim to be {8, 800}
        // but got {2}

        // std::cout << "\t reshaped to : ";
        // for(int i=0; i<shape->getTensor()->getNDim(); i++) {
        //     size_t d = shape->getTensor()->getDim(i);
        //     std::cout << d << " ";
        // }
        // std::cout << "\n";

        opNode->exlinkUpperNode(in);
        Tensor *inT;
        if(in->getTensor()->getNDim() == 4) {
            opNode->setOp(new TransposeOp(NHWC2NCHW));
            inT = new Tensor(in->getTensor()->getShuffledTensorShape(NHWC2NCHW));
            
        }

        std::string res_name = op.output(0);
        auto *out_tnode = new TensorNode(res_name, inT, opNode);
        name_tNode_map_[res_name] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }

    else if (opType == "Softmax") {
        auto *in = name_tNode_map_[op.input(0)]; // TensorNode<Dtype>*

        opNode = new OpNode(opName, new MatrixSoftmaxOp());
        opNode->exlinkUpperNode(in);
        
        std::string res_name = op.output(0);
        auto *tshape = in->getTensor()->getTensorShape();
        auto *out_tnode = new TensorNode(res_name, new Tensor(tshape), opNode);
        name_tNode_map_[opName] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }

    // !!! last but not least
    name_opNode_map_[opName] = opNode;
    graph_->pushOpNode(opNode);
}
} // namespace swc