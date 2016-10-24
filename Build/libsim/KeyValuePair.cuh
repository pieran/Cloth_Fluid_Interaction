#pragma once

#include <host_defines.h>

#ifndef uint
typedef unsigned int uint;
#endif

typedef struct __align__(8) {
	uint key;
	uint value;
} KeyValuePair;