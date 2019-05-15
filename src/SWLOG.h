/*************************************************************************
	> File Name: swlog.h
	> Author: cryinlaugh
	> Mail: cryinlaugh@gmail.com
	> Created Time: Tue 11 Dec 2018 10:08:09 AM UTC
 ************************************************************************/

#ifndef _SWLOG_H
#define _SWLOG_H

#include <cstdio>
#include <ctime>
#include <iostream>
#include <sstream>

static time_t t = time(0);
static struct tm ctm;
static std::ostringstream __oss;
static std::ostream *__os;

#define SWLOG_INFO SWLOG(cout, INFO)

#define SWLOG_ERROR SWLOG(cerr, ERROR)

// #define SWLOG_DEBUG SWLOG(cout, DEBUG)

#ifdef DEBUG
#define SWLOG(stream, type)                                                    \
    localtime_r(&t, &ctm);                                                     \
    std::stream << "[" #type "] [" << ctm.tm_year + 1900 << "-"                \
                << ctm.tm_mon + 1 << "-" << ctm.tm_mday << " " << ctm.tm_hour  \
                << ":" << ctm.tm_min << ":" << ctm.tm_sec << "] [" << __FILE__ \
                << ":" << __FUNCTION__ << ":" << __LINE__ << "] "

#else
#define SWLOG(stream, type)                                                    \
    __oss.clear();                                                             \
    __oss
#endif

// enabled only when
// 1. cmake -DLEVELDEBUG=xx ... and
// 2. LEVELDEBUG SWLOG_DEBUG(i)<<xxx; i >
#ifdef LEVELDEBUG
#define SWLOG_DEBUG(level)                                                     \
    __os = (level < LEVELDEBUG) ? &__oss : &std::cout;                         \
    (*__os).clear();                                                           \
    localtime_r(&t, &ctm);                                                     \
    *__os << "[DEBUG-" #level "] [" << ctm.tm_year + 1900 << "-"               \
          << ctm.tm_mon + 1 << "-" << ctm.tm_mday << " " << ctm.tm_hour << ":" \
          << ctm.tm_min << ":" << ctm.tm_sec << "] [" << __FILE__ << ":"       \
          << __FUNCTION__ << ":" << __LINE__ << "] "

#else
#define SWLOG_DEBUG(level)                                                     \
    __oss.clear();                                                             \
    __oss
#endif

#endif
