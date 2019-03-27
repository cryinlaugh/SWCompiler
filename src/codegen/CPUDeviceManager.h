/*************************************************************************
	> File Name: CPUDeviceManager.h
	> Author: wayne
	> Mail:  
	> Created Time: 四  3/14 15:49:39 2019
 ************************************************************************/
#ifndef _CPU_DEVICE_MANAGER_H_
#define _CPU_DEVICE_MANAGER_H_
#include "DeviceManager.h"
class CPUDeviceManager : public DeviceManager{

public:
    CPUDeviceManager(std::string name) : DeviceManager(DeviceType::CPU, name) {}
    ~CPUDeviceManager() {}
};
#endif
