/*************************************************************************
	> File Name: NVDeviceManager.h
	> Author: wayne
	> Mail:  
	> Created Time: 四  3/14 16:00:49 2019
 ************************************************************************/
#ifndef _NV_DEVICE_MANAGER_H_
#define _NV_DEVICE_MANAGER_H_
#include "DeviceManager.h"
class NVDeviceManager : public DeviceManager {

  public:
    NVDeviceManager(std::string name) : DeviceManager(DeviceType::CPU, name) {}
    ~NVDeviceManager() {}
};
#endif
