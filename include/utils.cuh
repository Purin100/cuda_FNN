/*
This file defines some misc stuff like functions, marcos, etc.
*/
#pragma once

#include <string.h>

/*string comparasion function for different platforms*/
#if defined(_WIN64) || defined(_WIN32)
#define mystrcmp(a,b) _stricmp(a,b)
#endif/*Windows*/
#ifdef linux
#define mystrcmp(a,b) strcasecmp(a, b)
#endif/*Linux*/

#define RELEASE(x) { if(x){delete[] x;x = nullptr;} }

#define BLOCK 512 // for im2col and col2im

typedef double MYTYPE;//data type for numbers in matrix, vector, and other classes in the project
typedef unsigned int UINT;