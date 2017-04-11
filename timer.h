#ifndef __TIMER_H__
#define __TIMER_H__

#include <sys/time.h>
#include <cstddef>

extern struct timeval timerStart;

inline void StartTimer()
{
 
	gettimeofday(&timerStart, NULL);
 
}

// time elapsed in ms
inline double GetTimer()
{
 
	struct timeval timerStop, timerElapsed;
	gettimeofday(&timerStop, NULL);
	timersub(&timerStop, &timerStart, &timerElapsed);
	return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
 
}

#endif // __TIMER_H__