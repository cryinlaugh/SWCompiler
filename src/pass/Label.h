/*************************************************************************
	> File Name: Label.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Wed 12 Dec 2018 08:39:10 AM UTC
 ************************************************************************/

#ifndef _LABEL_H
#define _LABEL_H

#include <string>

namespace swc{

class Label{
private:
    //init labeling
    std::string _nodeNameLabel;
    std::string _typeNameLabel;

    int _toLower;
    
public:
    Label() : _toLower(0){};
    ~Label(){};
    
    void setNodeNameLabel(std::string s) { _nodeNameLabel = s; };
    void setTypeNameLabel(std::string s) { _typeNameLabel = s; };

    void setLowerMark() { _toLower = 1;};

    std::string getNodeNameLabel() const { return _nodeNameLabel; };
    std::string getTypeNameLabel() const { return _typeNameLabel; };

    int getLowerMark() const { return _toLower;};


};

}
#endif
