/*************************************************************************
	> File Name: common.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 04 Dec 2018 08:09:21 AM UTC
 ************************************************************************/

#ifndef _COMMON_H
#define _COMMON_H

template <typename Dtype>
class SWMem{
private:
    size_t _len;
    Dtype* _data;
public:
    SWMem(size_t len, Dtype* data);
    ~SWMem();
    
    Dtype* data();
    Dtype* mutable_data();
};

#endif
