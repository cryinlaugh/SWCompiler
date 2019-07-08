#include "../caffe2.pb.h"

/*
net{
    op {
        name: "weight0"
        arg {
            name: "values"
            floats: 1.2
            floats: 1.5
            ...
        }
    }...
    arg{
        name: "iter"
        i: 500
    }...
}
*/

#define MAX_PROTO_SIZE 0x7FFFFFFF // 2G
caffe2::OperatorDef *addOp(caffe2::NetDef &net, std::string name, float *var,
                           size_t size) {
    caffe2::OperatorDef *op = net.add_op();
    op->set_name(name);
    caffe2::Argument *op_v = op->add_arg();
    op_v->set_name("values");
    for (size_t i = 0; i < size; i++)
        op_v->add_floats(var[i]);

    return op;
}

void loadFromSnapShot(caffe2::NetDef &net, std::string name, float *var,
                      size_t size) {
    /*
    caffe2::OperatorDef *op = net.add_op();
    op->set_name(name);
    caffe2::Argument *op_v = op->add_arg();
    op_v->set_name("values");
    for(size_t i=0; i<size; i++)
        op_v->add_floats(var[i]);
    */

    for (auto op : net.op()) {
        if (op.name() == name) {
            // std::cout << op.name() << std::endl;
            const caffe2::Argument values = op.arg(0);
            // std::cout << values.name() << std::endl;
            size_t k = 0;
            for (auto f : values.floats()) {
                var[k++] = f;
            }
            // std::cout << k << "/" << size << "\n";
            std::cout << name << ": " << size << " bytes.";
        }
    }
}
size_t getIterFromSnapShot(caffe2::NetDef &net) {
    for (auto &arg : net.arg()) {
        if (arg.name() == "iter") {
            size_t iter = arg.i();
            return iter;
        }
    }

    return 0;
}
