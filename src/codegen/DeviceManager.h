/*************************************************************************
	> File Name: DeviceManager.h
	> Author: wayne
	> Mail:  
	> Created Time: 四  3/14 15:24:17 2019
 ************************************************************************/
#ifndef _DEVICE_MANAGER_H_
#define _DEVICE_MANAGER_H_
#include <string>
#include "common.h"

class DeviceManager{
protected:
    DeviceType type_;
    std::string name_;
public:
    DeviceManager(DeviceType type, std::string name) : type_(type), name_(name) {} 
    virtual ~DeviceManager() {}
};
#endif
