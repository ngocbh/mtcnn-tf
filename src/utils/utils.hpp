#ifndef __UTILS_H__
#define __UTILS_H__

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <cassert>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include <sys/time.h>

using namespace std;

#define FOR(i,a) for (__typeof((a).begin()) i = (a).begin(); i != (a).end(); ++ i)

const double EPS = 1e-8;

inline bool myAssert(bool flg, string msg)
{
	if (!flg) {
		cerr << msg << endl;
		// exit(-1);
	}
	return flg;
}

template<class T>
inline void fromString(const string &s, T &x)
{
	stringstream in(s);
	in >> x;
}

unsigned long get_cur_time(void)
{
	struct timeval tv;
	unsigned long ts;

	gettimeofday(&tv,NULL);

	ts=tv.tv_sec*1000000+tv.tv_usec;

	return ts;
}

#endif
