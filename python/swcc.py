from ctypes import *
lib = CDLL("libswcc.so")
class OpNode(object):
    def __init__(self, val, optype, *parlist):
        lib.OpNode.argtypes = [c_char_p, c_char_p]
        lib.OpNode.restype = c_void_p

        lib.OpNode_toString.argtypes = [c_void_p, c_char_p]
        lib.OpNode_toString.restype = c_char_p

        lib.OpNode_link.argtypes = [c_void_p, c_void_p]
        
        self.obj = lib.OpNode(val, optype)
        for par in parlist:
            if isinstance(par, TensorNode):
                lib.OpNode_link(self.obj, par.obj)

    def toString(self):
        s = create_string_buffer('\0'*100)
        lib.OpNode_toString(self.obj, s)
        return s.value

class TensorShape(Structure):
    _fields_ = [("ndim", c_int),
                ("shape", POINTER(c_int64))]

class TensorNode(object):
    def __init__(self, val, dims, *oplist):
        lib.TensorNode.argtypes = [c_char_p, c_int64, POINTER(c_int64)]
        lib.TensorNode.restype = c_void_p
        
        lib.TensorNode_link.argtypes = [c_void_p, c_void_p]

        lib.TensorNode_toString.argtypes = [c_void_p, c_char_p]
        lib.TensorNode_toString.restype = c_char_p
        
        # _shape = TensorShape()
        # _shape.ndim = len(dims)
        # _shape.shape = (c_int64*len(dims))(dims)
        shape = (c_int64*len(dims))(*dims)
        self.obj = lib.TensorNode(val, len(dims), shape)
        for op in oplist:
            if isinstance(op, OpNode):
                lib.TensorNode_link(self.obj, op.obj)

    def toString(self):
        s = create_string_buffer('\0'*100)
        lib.TensorNode_toString(self.obj, s)
        return s.raw

class IRGraph(object):
    def __init__(self, val=None):
        lib.IRGraph.restype = c_void_p
        self.obj = lib.IRGraph(val)
        self.__ops = []
        self.__tensors = []

        lib.IRGraph_addOpNode.argtypes=[c_void_p, c_void_p]
        lib.IRGraph_addTensorNode.argtypes=[c_void_p, c_void_p]
        lib.IRGraph_summary.argtypes = [c_void_p]
        lib.IRGraph_summary.restype = c_char_p
        lib.IRGraph_dotGen.argtypes = [c_void_p, c_char_p]

    def addOpNode(self, op):
        lib.IRGraph_addOpNode(self.obj, op.obj)
        self.__ops.append(op)

    def addTensorNode(self, t):
        lib.IRGraph_addTensorNode(self.obj, t.obj)
        self.__tensors.append(t)

    def add(self, *nodes):
        for node in nodes:
            if isinstance(node, TensorNode):
                self.addTensorNode(node)
            else:
                self.addOpNode(node)

    def dump(self):
        """
        Dump neural network graph.
        OpNode: [name]
          op: [OpType]
          nInput: in(parent) nodes
          nOutput: out(children) nodes
        ...
        TensorNode:
          tensorDim: n
          x_0 x_1 ... x_(n-1)
        """
        for op in self.__ops:
            print (op.toString())
        for tensor in self.__tensors:
            print (tensor.toString())
            
    def summary(self):
        return lib.IRGraph_summary(self.obj)
    
    def dotGen(self, path):
        lib.IRGraph_dotGen(self.obj, path)
