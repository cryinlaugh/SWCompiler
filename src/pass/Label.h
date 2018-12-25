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
    std::string _typeNameLabel;
    
public:
    Label(){};
    ~Label(){};
    
    void setTypeNameLabel(std::string s) { _typeNameLabel = s; };

    std::string getTypeNameLabel() const { return _typeNameLabel; };

};

}
#endif
