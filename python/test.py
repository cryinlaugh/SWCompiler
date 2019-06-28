from swcc import *

nn = IRGraph()

data0 = TensorNode("data0", (8,784))
weight0 = TensorNode("weight0", (784,512))
bias0 = TensorNode("bias0", (512,))
fc0 = OpNode("fc0", "FC", data0, weight0, bias0)
data1 = TensorNode("data1", (8,512), fc0)

tanh0 = OpNode("tanh0", "Tanh", data1)
data2 = TensorNode("data2", (8, 512), tanh0)

weight1 = TensorNode("weight1", (512,10))
bias1 = TensorNode("bias1", (10,)) 
fc1 = OpNode("fc1", "FC", data2, weight1, bias1)
data3 = TensorNode("data3", (8,10), fc1)

softmax = OpNode("softmax", "Softmax", data3)
data4 = TensorNode("data4", (8,10), softmax)

# MXNet-like
nn.add(
    data0, weight0, bias0,
    fc0, data1,
    tanh0, data2,
    weight1, bias1,
    fc1, data3,
    softmax, data4
)

# Or you can add nodes like this
# nn.addTensorNode(data0)
# nn.addTensorNode(weight0)
# nn.addTensorNode(bias0)
# nn.addOpNode(fc0)
# nn.addTensorNode(data1)
# nn.addOpNode(tanh0)
# nn.addTensorNode(data2)
# nn.addTensorNode(weight1)
# nn.addTensorNode(bias1)
# nn.addOpNode(fc1)
# nn.addTensorNode(data3)
# nn.addOpNode(softmax)
# nn.addTensorNode(data4)

print (nn.summary())
print (fc0.toString())
print (weight1.toString())
nn.dump()
nn.dotGen("nn.dot")