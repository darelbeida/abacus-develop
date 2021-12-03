/*
 * Author: Xiaoying Jia
 * Project: Faster DFT
 * Department: ByteDance Data-AML
 * Email: {jiaxiaoying}@bytedance.com
 */
#pragma once
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <cmath>
using namespace std;

double diffTime(timeval start, timeval end)
{
    return (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) * 0.001;
}


