/***********************************************
#
#      Filename: src/devices/DeviceTopology.h
#
#        Author: whl - lethewang@yeah.net
#   Description: ---
#        Create: 2019-09-20 10:43:09
# Last Modified: 2019-09-20 10:43:09
***********************************************/
#include<vector>
namespace swc{

class ComputeDeivce;
class CommunicateDevice;
class DeviceConfig;
class DeviceTopology{

public:
    DeviceTopology();
    ~DeviceTopology();
private:
    std::vector<ComputeDeivce> computeDevices;
    std::vector<CommunicateDevice> communicateDevices;
};

}
