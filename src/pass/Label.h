/*************************************************************************
	> File Name: Label.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Wed 12 Dec 2018 08:39:10 AM UTC
 ************************************************************************/

#ifndef _LABEL_H
#define _LABEL_H

#include <string>

#include "common.h"

namespace swc{

class Label{
private:
    //init labeling
    std::string _nodeNameLabel;
    std::string _typeNameLabel;

    int _toLower;

    TensorInitType _initTypeLabel;
    
public:
<<<<<<< HEAD
    Label() : _toLower(0), _initTypeLabel(TensorInitType::NONE){};
    ~Label(){};
=======
    Label() : _toLower(0){};
    virtual ~Label(){};
>>>>>>> IRDesign

    void destroy(){
        this->~Label();
    };

    void setNodeNameLabel(std::string s) { _nodeNameLabel = s; };
    void setTypeNameLabel(std::string s) { _typeNameLabel = s; };
    void setTensorInitTypeLabel(TensorInitType type) { _initTypeLabel = type; };

    void setLowerMark() { _toLower = 1;};

    std::string getNodeNameLabel() const { return _nodeNameLabel; };
    std::string getTypeNameLabel() const { return _typeNameLabel; };
    TensorInitType getTensorInitTypeLabel() const { return _initTypeLabel; };

    int getLowerMark() const { return _toLower;};


};

}
#endif
