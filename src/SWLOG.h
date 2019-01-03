/*************************************************************************
	> File Name: swlog.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 11 Dec 2018 10:08:09 AM UTC
 ************************************************************************/

#ifndef _SWLOG_H
#define _SWLOG_H

#include <ctime>
#include <cstdio>
#include <iostream>

static time_t t = time(0) ;
static struct tm ctm; 

#define SWLOG_INFO SWLOG(cout, INFO)

#define SWLOG_ERROR SWLOG(cerr, ERROR)

#define SWLOG_DEBUG SWLOG(cout, DEBUG)


#define SWLOG(stream, type) \
    localtime_r(&t, &ctm); \
    std::stream<<"["#type"] ["\
             << ctm.tm_year + 1900 << "-" \
             << ctm.tm_mon + 1 << "-" \
             << ctm.tm_mday << " " \
             << ctm.tm_hour << ":" \
             << ctm.tm_min << ":" \
             << ctm.tm_sec << "] [" \
             << __FILE__ << ":" \
             << __FUNCTION__ << ":" \
             << __LINE__ << "] "

#endif
