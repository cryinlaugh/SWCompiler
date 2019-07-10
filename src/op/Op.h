/*************************************************************************
	> File Name: Op.h
	> Author: cryinlaugh 
	> Mail: cryinlaugh@gmail.com 
	> Created Time: 二 12/ 4 15:57:08 2018
 ************************************************************************/

#ifndef _OP_H
#define _OP_H

#include <string>
#include <unordered_map>

#include "SWLOG.h"
#include "common.h"

namespace swc {

// Forward declarations
class Tensor;
class IRGraph;
class IRNode;

namespace op {

class Op {
  public:
    Op(OpType opType = BASIC_OP, int nInput = 0, int nOutput = 0,
       std::string opClassName = NULL)
        : _opType(opType), _nInput(nInput), _nOutput(nOutput),
          _opClassName(opClassName) {

        _nInputTensor = 0;
        _nOutputTensor = 0;
    };

    ~Op(){};

    virtual void destroy(){};
    virtual void autoDiff(IRGraph* graph,
            IRNode* opNode,
            std::unordered_map<IRNode*, IRNode*> &gradNodeMap){};
    virtual void einsumLowering(IRGraph *graph, IRNode *node) {
        SWLOG_DEBUG(100) << "EinsumLowering unimplemented in base Op class" << std::endl;
    }

    void addInputTensor(Tensor *inputTensor) {
        _inputTensors.push_back(inputTensor);
        _nInputTensor++;
    }

    void addOutputTensor(Tensor *outputTensor) {
        _outputTensors.push_back(outputTensor);
        _nOutputTensor++;
    }
    bool check();

    OpType getOpType() { return _opType; }

    inline const std::string getOpName() { return _opClassName; }

    inline int getnInput() { return _nInput; }
    inline int getnOutput() { return _nOutput; }

    // for lowering
    virtual void lowering(IRGraph *graph, IRNode *node) {
        SWLOG_DEBUG(100) << "Lowering unimplemented in base Op class" << std::endl;
    }

    /*
    Op *clone() const {
        return new Op(_opType, _nInput, _nOutput, _opClassName);
    }
    */

    inline int getInputDims(int n) { return _inputNDims[n]; }
    inline int getOutputDims(int n) { return _outputNDims[n]; }

  protected:
    /* The following variables are constant values in a specific Op Class
       indicating what kind of input/output tensors it should keep. */

    const OpType _opType;          // enum var, define the type of operation
    const int _nInput;             // nums of input  tensor
    const int _nOutput;            // nums of output tensor
    std::vector<int> _inputNDims;  // input  tensors
    std::vector<int> _outputNDims; // output tensors

    const std::string _opClassName;

    /* The following variables indicating the real input/output tensors
       that the Op really have, its useful in analyses or ref-code-generation.
     */

    int _nInputTensor;
    int _nOutputTensor;
    std::vector<Tensor *> _inputTensors;
    std::vector<Tensor *> _outputTensors;

    /* Edit by zwl @ 20190705
     * The following variables indicate the dimension-level relation shape of input/output tensors,
     * which is designed as a generalized abstraction for tensors (similar to a Einsum expression),
     * and is used for analyzing and generating parallel strategies for a tensor graph.
     * 
     * _einOp is a label variable indicats whether the Op can be discribed using Einsum expression.
     *
     */
    int _einOp;
    std::vector<std::string> _einRep;

    friend class ParallelGen;
};

} // namespace op
} // namespace swc

#endif
