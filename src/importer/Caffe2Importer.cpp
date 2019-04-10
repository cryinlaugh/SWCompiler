/*************************************************************************
	> File Name: Caffe2Importer.cpp
	> Author: wayne
	> Mail:  
	> Created Time: 四  3/28 10:12:26 2019
 ************************************************************************/

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cstdlib>
#include "Caffe2Importer.h"
#include "caffe2.pb.h"
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "graphIR/IRNode.h"
#include "graphIR/TensorNode.h"
#include "graphIR/OpNode.h"
#include "tensor/tensor.h"
#include "op/Op.h"
#include "op/dlOp/dlOp.h"
#include "common.h"
#include "SWDSL.h"

namespace swc{

using ArgumentMap = std::unordered_map<std::string, const caffe2::Argument*>; 

static std::vector<size_t> getPads(const ArgumentMap &args) {
    if(args.count("pad")){
        int pad = args.at("pad")->i();
        std::vector<size_t> pads(4, pad);
        return pads;
    }

    if(args.count("pad_t")){
        size_t p_t = args.at("pad_t")->i(); 
        size_t p_l = args.at("pad_l")->i(); 
        size_t p_b = args.at("pad_b")->i(); 
        size_t p_r = args.at("pad_r")->i(); 
        std::vector<size_t> pads({p_t, p_l, p_b, p_r});
        return pads;
    }

    if(args.count("pads")){
        std::vector<size_t> pads;
        for(auto i : args.at("pads")->ints())
            pads.push_back(i);
        return pads;
    }
   return {0, 0, 0, 0}; 
}

// kernel for Conv and Pooling
static std::vector<size_t> getKernels(const ArgumentMap &args){
    if(args.count("kernel")){
        int value = args.at("kernel")->i();
        std::vector<size_t> kernels(2, value);
        return kernels;
    } 
    // TODO
    //kernel_h kernel_w ...
    return {0, 0};
}
static std::vector<size_t> getStrides(const ArgumentMap &args){
    if(args.count("stride")){
        int value = args.at("stride")->i();
        std::vector<size_t> strides(2, value);
        return strides;
    } 
    // TODO
    //kernel_h kernel_w ...
    return {1, 1};
}

static std::vector<size_t> inferConvOutDims(size_t ih, size_t iw, std::vector<size_t> &kernels, std::vector<size_t> &strides,
   std::vector<size_t> &pads){
    assert(kernels.size() == 2);
    assert(strides.size() == 2);
    assert(pads.size() == 4);

    size_t oh = ((ih+pads[0]+pads[2]-kernels[0])/strides[0] + 1);
    size_t ow = ((iw+pads[1]+pads[3]-kernels[1])/strides[1] + 1);
    return {oh, ow};
}

static std::vector<size_t> inferTransOutDims(std::vector<size_t> &idims,   std::vector<size_t> &shuffle){
    // TODO check illegal shuffle index 
    std::vector<size_t> odims;
    for(auto idx : shuffle){
        if(idx < idims.size())
            odims.push_back(idims.at(idx));
    } 
    return odims;
}

static std::string getNodeName(std::string oldName){
    assert(!oldName.empty() && "inputName empty");
    std::string name;
    for(const char c : oldName){
        if(c=='/' || c=='.' || c=='-')
            name.push_back('_');
        else
            name.push_back(c);
    }

    return name;
}

// TODO: as TensorShape member function
TensorShape* getShuffledTensorShape(const TensorShape * in,
    const std::vector<size_t> &shuffle){
    std::vector<unsigned long> * shape = new std::vector<unsigned long>();
    for(auto idx : shuffle){
        if((int)idx < in->getNDim())
            shape->push_back(in->getDim(idx));
    } 

    return new TensorShape(shape);
}

Caffe2Importer::Caffe2Importer(IRGraph *g, const std::string &netProtoFile, const std::string &tensorProtoFile, 
    std::vector<TensorNode*> &udef_nodes) {
    graph_ = g;

    for(auto tnode : udef_nodes){
        graph_->pushTensorNode(tnode);
        std::string name = tnode->name();

        name_tNode_map_[name] = tnode;
    }
    
    caffe2::NetDef tensors;
    caffe2::NetDef network;
    loadProto(network, netProtoFile);
    loadProto(tensors, tensorProtoFile);
    
    loadTensors(tensors);
    loadNetwork(network);
    std::cout << "ctor of Caffe2Importer\n";
}

void Caffe2Importer::loadProto(caffe2::NetDef &net, const std::string &filename){
    std::ifstream ff(filename, std::ios::in | std::ios::binary);
    assert(ff && "Can't find the model or network files.");

    bool parseNet = false;
    if (filename.find(".pbtxt") != std::string::npos) {
        std::string str((std::istreambuf_iterator<char>(ff)),
                    std::istreambuf_iterator<char>());
        parseNet = google::protobuf::TextFormat::ParseFromString(str, &net);
    } else {
        // Construct and configure a Coded Input Stream
        google::protobuf::io::IstreamInputStream filestr(&ff);
        google::protobuf::io::CodedInputStream codedstr(&filestr);
        // Don't warn about large file sizes.
        codedstr.SetTotalBytesLimit(MAX_PROTO_SIZE, MAX_PROTO_SIZE);
        parseNet = net.ParseFromCodedStream(&codedstr);
  }

    assert(parseNet && "Failed to parse the network descriptor.");
}
void Caffe2Importer::loadNetwork(caffe2::NetDef &net) {
    if(net.has_name()){
        std::cout << "loading network " << net.name() << std::endl;
    }
    for(auto &op : net.op()){
        loadOp(op);
    }
}
void Caffe2Importer::loadOp(const caffe2::OperatorDef &op){
    std::string opType = op.type(); 
    // std::string opName = op.name().length() ? op.name() : op.output(0);
    //
    std::string opName = opType;
    transform(opName.begin(), opName.end(), opName.begin(), ::tolower);

    std::cout << opName<< " " << op.output(0) << std::endl
    << "\ttype  : " << op.type() << std::endl
    << "\tinput : " << op.input_size() << std::endl
    << "\toutput: " << op.output_size() << std::endl;     


    // get caffe2::Argument map
    std::unordered_map<std::string, const caffe2::Argument*> args;
    for(auto &arg : op.arg()){
        assert(arg.has_name() && "Argument without name!");
        args[arg.name()] = &arg; 
    }
    
    OpNode *opNode;
    if(opType == "Conv"){
        
        // assert(op.input_size() == 3 && "conv bias is needed!!");
        auto data = name_tNode_map_[op.input(0)]; 
        auto weight = name_tNode_map_[op.input(1)];
        
        std::vector<unsigned long> inDims = data->getDims();

        TensorNode * bias;
        if(op.input_size() == 3){
            bias = name_tNode_map_[op.input(2)];
        }else{
            std::string nm = opName+"_bias";
            bias  = new TensorNode(nm.c_str(), {(int)inDims[3]});
            graph_->pushTensorNode(bias);
        }

        std::vector<size_t> kernels = getKernels(args);
        std::vector<size_t> strides = getStrides(args);
        std::vector<size_t> pads = getPads(args);

        std::string trans_op_name = "op_" + weight->name() + "_T";
        auto trans = new OpNode(trans_op_name.c_str(), new TranposeOp(NCHW2NHWC)); 
        LINKUPPER(trans, weight); 

        Tensor *wt = new Tensor(getShuffledTensorShape(weight->getTensor()->getTensorShape(),NCHW2NHWC)); 
        std::string trans_name = weight->name() + "_T";
        auto w_trans = new TensorNode(trans_name.c_str(), wt, trans);


        auto *convOp = new Conv2dOp(kernels, strides, pads);
        opNode = new OpNode(opName.c_str(), convOp); 
        opNode->exlinkUpperNode(data, w_trans, bias); 

        std::vector<size_t> ohw = inferConvOutDims(inDims[1], inDims[2], 
            kernels, strides, pads); 

        graph_->pushOpNode(trans);
        graph_->pushTensorNode(w_trans);

    

        int n = inDims[0]; // from data
        int c = w_trans->getDims()[0]; 

        std::string res_name = op.output(0);
        auto out_tnode = new TensorNode(res_name.c_str(), {n, (int)ohw[0], (int)ohw[1], c}, opNode); 
        name_tNode_map_[res_name] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }

    if(opType == "SpatialBN") {
        assert(op.input_size() == 5 && "SpatialBN need 5 intput!!");
        auto data = name_tNode_map_[op.input(0)]; 
        auto scale = name_tNode_map_[op.input(1)];
        auto bias = name_tNode_map_[op.input(2)];
        auto mean = name_tNode_map_[op.input(3)];
        auto var = name_tNode_map_[op.input(4)];

        float epsilon = 1e-5f;
        if(args.count("epsilon")){
            epsilon = args.at("epsilon")->f();
        }

        if(args.count("order")){
            assert((args.at("order")->s() == "NCHW") && "only support NCHW Caffe2 Model");
        }

        auto *BNOp = new BatchNormalizationOp(epsilon);
        opNode = new OpNode(opName.c_str(), BNOp); 
        opNode->exlinkUpperNode(data, scale, bias, mean, var); 

        std::string res_name = op.output(0);
        auto *tshape = data->getTensor()->getTensorShape();
        auto *out_tnode = new TensorNode(res_name.c_str(), new Tensor(tshape), opNode); 
        name_tNode_map_[res_name] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }
    
    if(opType == "Relu"){
        opNode = new OpNode(opName.c_str(), new ReluOp());

        std::string iname = op.input(0);
        auto*  in = name_tNode_map_[iname];
        LINKUPPER(opNode, in);

        std::string res_name = op.output(0);
        TensorNode *out_tnode;
        if(name_tNode_map_.count(res_name)){
            std::cout << res_name << " exist\n";
            auto *tensor = name_tNode_map_[res_name]->getTensor();
            out_tnode = new TensorNode(res_name.c_str(), tensor, opNode); 
        }else{
            std::cout << res_name << " non exist\n";
            auto *tshape = in->getTensor()->getTensorShape();
            out_tnode = new TensorNode(res_name.c_str(), new Tensor(tshape), opNode); 
        }
        // name mapping to newest operator's  res TensorNode
        name_tNode_map_[res_name] = out_tnode; 
        graph_->pushTensorNode(out_tnode);
    }

    if(opType == "Sum"){
        opNode = new OpNode(opName.c_str(), new ElementAddOp());
        auto*  lhs = name_tNode_map_[op.input(0)];
        auto*  rhs = name_tNode_map_[op.input(1)];
        LINKUPPER(opNode, lhs, rhs);

        std::string res_name = op.output(0);
        TensorNode *out_tnode;
        if(name_tNode_map_.count(res_name)){
            auto *tensor = name_tNode_map_[res_name]->getTensor();
            out_tnode = new TensorNode(res_name.c_str(), tensor, opNode); 
        }else{
            auto *tshape = lhs->getTensor()->getTensorShape();
            out_tnode = new TensorNode(res_name.c_str(), new Tensor(tshape), opNode); 
        }
        // name mapping to newest operator's  res TensorNode
        name_tNode_map_[res_name] = out_tnode; 
        graph_->pushTensorNode(out_tnode);
    }

    if(opType == "MaxPool"){
        auto in = name_tNode_map_[op.input(0)]; 
        
        std::vector<size_t> kernels = getKernels(args);
        std::vector<size_t> strides = getStrides(args);
        std::vector<size_t> pads = getPads(args);

        auto *poolOp = new MaxPoolOp(kernels, strides, pads);
        opNode = new OpNode(opName.c_str(), poolOp);
        LINKUPPER(opNode, in);

        std::vector<size_t> inDims = in->getDims();
        int n = inDims[0];
        int c = inDims[3];
        std::vector<size_t> ohw = inferConvOutDims(inDims[1], inDims[2], 
            kernels, strides, pads); 

        std::string res_name = op.output(0);
        auto *out_tnode = new TensorNode(res_name.c_str(), {n, (int)ohw[0], (int)ohw[1], c}, opNode); 
        name_tNode_map_[res_name] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }

    if(opType == "AveragePool"){
        auto in = name_tNode_map_[op.input(0)]; 
        
        std::vector<size_t> kernels = getKernels(args);
        std::vector<size_t> strides = getStrides(args);
        std::vector<size_t> pads = getPads(args);

        auto *poolOp = new AvgPoolOp(kernels, strides, pads);
        opNode = new OpNode(opName.c_str(), poolOp);
        LINKUPPER(opNode, in);

        std::vector<size_t> inDims = in->getDims();
        int n = inDims[0];
        int c = inDims[3];
        std::vector<size_t> ohw = inferConvOutDims(inDims[1], inDims[2], 
            kernels, strides, pads); 
        std::cout << "AveragePool "
                    << n << " " << ohw[0] << " " << ohw[1] << " " << c << "\n";

        std::string res_name = op.output(0);
        auto *out_tnode = new TensorNode(res_name.c_str(), {n, (int)ohw[0], (int)ohw[1], c}, opNode); 
        name_tNode_map_[res_name] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }

    if(opType == "FC"){
        opNode = new OpNode(opName.c_str(), new MatrixMatrixFCOp());
        
        assert(op.input_size() == 3 && "FC bias is needed!!");
        auto in = name_tNode_map_[op.input(0)]; 
        auto weight = name_tNode_map_[op.input(1)];
        auto bias = name_tNode_map_[op.input(2)];
        
        // trans weight
        std::string trans_op_name = "op_" + weight->name() + "_T";
        auto trans = new OpNode(trans_op_name.c_str(), new TranposeOp({1, 0})); 
        LINKUPPER(trans, weight); 

        Tensor *wt = new Tensor(getShuffledTensorShape(weight->getTensor()->getTensorShape(),{1, 0})); 
        std::string trans_name = weight->name() + "_T";
        auto w_trans = new TensorNode(trans_name.c_str(), wt, trans);


        if(in->getTensor()->getNDim() == 4){
            // trans input
            
            std::string trans_in_op_name = "op_" + in->name() + "_T";
            
            auto trans_in = new OpNode(trans_in_op_name.c_str(), new TranposeOp(NHWC2NCHW)); 
            LINKUPPER(trans_in, in); 
            
            Tensor *inT = new Tensor(getShuffledTensorShape(in->getTensor()->getTensorShape(),NHWC2NCHW)); 
            std::string trans_in_name = in->name() + "_T";
            auto in_trans = new TensorNode(trans_in_name.c_str(), inT, trans_in);
            opNode->exlinkUpperNode(in_trans, w_trans, bias); 
            graph_->pushOpNode(trans, trans_in);
            graph_->pushTensorNode(w_trans, in_trans);
        }else{
            opNode->exlinkUpperNode(in, w_trans, bias); 
            graph_->pushOpNode(trans);
            graph_->pushTensorNode(w_trans);
        }


        std::vector<unsigned long> inDims = in->getDims();
        std::vector<unsigned long> weightDims = w_trans->getDims();
        size_t flatdim =  1;
        for(size_t i=1; i< inDims.size(); i++)
            flatdim *= inDims[i];

        std::cout << "flatdim=" << flatdim << " weightDims[0]=" << weightDims[0] << "\n";
        assert((flatdim == weightDims[0]) && "input flattenedDim not equal with weight\n");

        int n = inDims[0]; // from data
        int sec = weightDims[1];

        std::string res_name = op.output(0);
        auto *out_tnode = new TensorNode(res_name.c_str(), {n, sec}, opNode); 
        name_tNode_map_[res_name] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }
    
    if(opType == "Softmax"){ 
        std::string iname = op.input(0);
        auto *in = name_tNode_map_[iname]; // TensorNode<Dtype>*

        opNode = new OpNode(opName.c_str(), new MatrixSoftmaxOp());
        LINKUPPER(opNode, in);

        std::string res_name = op.output(0);
        auto *tshape = in->getTensor()->getTensorShape();
        auto *out_tnode = new TensorNode(res_name.c_str(), new Tensor(tshape), opNode); 
        name_tNode_map_[opName] = out_tnode;
        graph_->pushTensorNode(out_tnode);
    }

    name_opNode_map_[opName] = opNode;  
    graph_->pushOpNode(opNode);
}

void Caffe2Importer::loadTensors(caffe2::NetDef &tensors){
    for(auto &tensor : tensors.op())
        loadTensor(tensor);
}

void Caffe2Importer::loadTensor(const caffe2::OperatorDef &op){

    Tensor *tensor = new Tensor();

    const std::string type = op.type();
    if(type == "GivenTensorFill"){
    // op(tensor) name() probably null
    // but output is directly related to op who consume it

        // get caffe2::Argument map
        std::unordered_map<std::string, const caffe2::Argument*> args;
        for(auto &arg : op.arg()){
            assert(arg.has_name() && "Argument without name!");
            args[arg.name()] = &arg; 
        }

        // get "shape"
        std::vector<unsigned long> *shape = new std::vector<unsigned long>();
        for(auto i : args["shape"]->ints())
           shape->push_back(i);
        tensor->reset(new TensorShape(shape));

        // get value
        std::vector<float> tensorValue;
        for(auto value : args["values"]->floats()){
            float v = value;
            tensorValue.push_back(v);
        }

        std::ostringstream address;
        address << (void const *)tensor;
        std::string path = "/tmp/SW/" + address.str();
        
        tensor->setTensorInit(TensorInitType::FILE, path);
        // mkstemp(&path[0]);
        std::ofstream fout(path, std::ios::out | std::ios::binary);
        fout.write((char*)&tensorValue[0], tensorValue.size() * sizeof(float));
        fout.close();
        tensorValue.clear();

        for(auto &output : op.output()){
            std::string name = output;
            std::cout << "tensor " << name << std::endl
                << "\tpath: " << path << std::endl
                << "\tdim : " << shape->size()<< std::endl
                << "\tsize: " << tensor->size() << std::endl;
        }

    }     

    if(type == "ConstantFill") {
        // get caffe2::Argument map
        std::unordered_map<std::string, const caffe2::Argument*> args;
        for(auto &arg : op.arg()){
            assert(arg.has_name() && "Argument without name!");
            args[arg.name()] = &arg; 
        }

        // get "shape"
        std::vector<unsigned long> *shape = new std::vector<unsigned long>();
        for(auto i : args["shape"]->ints())
           shape->push_back(i);
        tensor->reset(new TensorShape(shape));

        float constValue = (args.count("value") && args["value"]->has_f()) ? args["value"]->f() : 0.0f;
        tensor->setTensorInit(TensorInitType::CONSTANT, constValue);

        for(auto &output : op.output()){
            std::string name = output;
            std::cout << "tensor " << name << std::endl
                << "\tdim : " << shape->size()<< std::endl
                << "\tsize: " << tensor->size() << std::endl
                << "\tvalue:" << constValue << std::endl;
        }

    }

    if(type == "UniformFill") {
        // TODO
    }
    for(auto &output : op.output()){
        std::string name = output;
        if(name_tNode_map_.count(name))
            return;
        TensorNode *tnode = new TensorNode(name.c_str(), tensor);
        graph_->pushTensorNode(tnode);
        name_tNode_map_[name] = tnode;
    }
}

} // swc
