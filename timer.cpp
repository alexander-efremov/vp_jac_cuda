#include "timer.h"
#ifdef WIN32
__int64 timerStart = 0;
#else
struct timeval timerStart;
#endif