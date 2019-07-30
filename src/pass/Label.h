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

namespace swc {

class Label {
  private:
    // init labeling
    std::string _nodeNameLabel;
    std::string _typeNameLabel;

    int _toLower;

    int _train{0};

    // This label is to mark those nodes that can not be
    // eliminated even the out rank is zero;
    // 0 by default
    int _isOut;
    
    TensorInitType _initTypeLabel;
    Device _dev;

  public:
    Label() : _toLower(0), _isOut(0), _initTypeLabel(TensorInitType::NONE){};
    virtual ~Label(){};

    void destroy() { this->~Label(); };

    void setNodeNameLabel(std::string s) { _nodeNameLabel = s; }
    void setTypeNameLabel(std::string s) { _typeNameLabel = s; }
    void setTensorInitTypeLabel(TensorInitType type) { _initTypeLabel = type; }
    void setDeviceLabel(DeviceType type, int id = 0) {
        _dev.type = type;
        _dev.id = id;
    }

    void setLowerMark() { _toLower = 1; };
    int getLowerMark() const { return _toLower; }

    void setTraining(int train) { _train = train; }
    bool needTraining() { return _train == 1; }

    void setIsOut() { _isOut = 1; };
    int getIsOut() const { return _isOut; }
    
    std::string getNodeNameLabel() const { return _nodeNameLabel; }
    std::string getTypeNameLabel() const { return _typeNameLabel; }
    TensorInitType getTensorInitTypeLabel() const { return _initTypeLabel; }
    Device getDeviceLabel() const { return _dev; }
};

} // namespace swc
#endif
