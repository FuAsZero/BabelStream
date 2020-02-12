
// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code

#pragma once

#include <iostream>
#include <stdexcept>
#include <sstream>

#include "Stream.h"

#define IMPLEMENTATION_STRING "HIP"

#include <hip/hip_runtime.h>
#include "hip/hip_ext.h"

extern hipEvent_t start_ev;
extern hipEvent_t stop_ev;
extern float kernel_time;

template <class T>
class HIPStream : public Stream<T>
{
  protected:
    // Size of arrays
    unsigned int array_size;

    // timing mode
    unsigned int timing_mode;

    // Compute Unit number
    unsigned int cu_num;

    // Host array for partial sums for dot kernel
    T *sums;

    // Device side pointers to arrays
    T *d_a;
    T *d_b;
    T *d_c;
    T *d_sum;

#ifdef PURE_RDWR
    // Device side pointer to write kernel array
    T *d_d;

    // Host array for partial sums for read kernel
    T *sums_a;

    // Device side pointer to array for read kernel
    T *sum_a;
#endif

  public:

    HIPStream(const unsigned int, const int, const unsigned int, const unsigned int);
    ~HIPStream();

#ifdef PURE_RDWR
    virtual void write() override;
    virtual T read() override;
#endif

    virtual void copy() override;
    virtual void add() override;
    virtual void mul() override;
    virtual void triad() override;
    virtual T dot() override;

    virtual void init_arrays(T initA, T initB, T initC) override;
    virtual void read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c) override;
};
