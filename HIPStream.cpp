// Copyright (c) 2015-16 Tom Deakin, Simon McIntosh-Smith,
// University of Bristol HPC
//
// For full license terms please see the LICENSE file distributed with this
// source code


#include "HIPStream.h"
#include "hip/hip_runtime.h"

//#define TBSIZE 1024
#define TBSIZE		128			//use small TBSIZE can work for more flexible Array size

#ifdef MI50
#define DOT_NUM_BLOCKS  (256/64*60)		//MI50 has 60CU, while MI60 has 64CU
#else
#define DOT_NUM_BLOCKS  256
#endif

template <typename T>
__global__ void copy_kernel(const T * a, T * c);

template <typename T>
__global__ void mul_kernel(T * b, const T * c);

template <typename T>
__global__ void add_kernel(const T * a, const T *b, T * c);

template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c);

template <typename T>
__global__ void dot_kernel(const T * a, const T * b, T * sum, unsigned int array_size);

#ifdef PURE_RDWR
//#define READ_NUM_BLOCKS DOT_NUM_BLOCKS
#define WIDTH 4
#define READ_NUM_BLOCKS (array_size/(4*WIDTH*TBSIZE/sizeof(T)))

template <typename T>
__global__ void read_kernel(const T * a, T * sum, unsigned int array_size);

template <typename T>
__global__ void write_kernel(T * d);
#endif

void check_error(void)
{
  hipError_t err = hipGetLastError();
  if (err != hipSuccess)
  {
    std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
    exit(err);
  }
}

template <class T>
HIPStream<T>::HIPStream(const unsigned int ARRAY_SIZE, const int device_index)
{

  if (ARRAY_SIZE % TBSIZE != 0)
  {
    std::stringstream ss;
    ss << "Array size must be a multiple of " << TBSIZE;
    throw std::runtime_error(ss.str());
  }

  // Set device
  int count;
  hipGetDeviceCount(&count);
  check_error();
  if (device_index >= count)
    throw std::runtime_error("Invalid device index");
  hipSetDevice(device_index);
  check_error();

  // Print out device information
  std::cout << "Using HIP device " << getDeviceName(device_index) << std::endl;
  std::cout << "Driver: " << getDeviceDriver(device_index) << std::endl;

  array_size = ARRAY_SIZE;

  // Allocate the host array for partial sums for dot kernels
  sums = (T*)malloc(sizeof(T) * DOT_NUM_BLOCKS);

  // Check buffers fit on the device
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, 0);
  if (props.totalGlobalMem < 3*ARRAY_SIZE*sizeof(T))
    throw std::runtime_error("Device does not have enough memory for all 3 buffers");

  // Create device buffers
  hipMalloc(&d_a, ARRAY_SIZE*sizeof(T));
  check_error();
  hipMalloc(&d_b, ARRAY_SIZE*sizeof(T));
  check_error();
  hipMalloc(&d_c, ARRAY_SIZE*sizeof(T));
  check_error();
  hipMalloc(&d_sum, DOT_NUM_BLOCKS*sizeof(T));
  check_error();

#ifdef PURE_RDWR
  hipMalloc(&d_d, ARRAY_SIZE*sizeof(T));
  check_error();

  sums_a = (T*)malloc(sizeof(T) * READ_NUM_BLOCKS);

  hipMalloc(&sum_a, READ_NUM_BLOCKS * sizeof(T));
  check_error();
#endif
}


template <class T>
HIPStream<T>::~HIPStream()
{
  free(sums);

  hipFree(d_a);
  check_error();
  hipFree(d_b);
  check_error();
  hipFree(d_c);
  check_error();
  hipFree(d_sum);
  check_error();

#ifdef PURE_RDWR
  hipFree(d_d);
  check_error();
  free(sums_a);
  hipFree(sum_a);
  check_error();
#endif
}


template <typename T>
__global__ void init_kernel(T * a, T * b, T * c, T initA, T initB, T initC)
{
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  a[i] = initA;
  b[i] = initB;
  c[i] = initC;
}

template <class T>
void HIPStream<T>::init_arrays(T initA, T initB, T initC)
{
  hipLaunchKernelGGL(HIP_KERNEL_NAME(init_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_a, d_b, d_c, initA, initB, initC);
  check_error();
  hipDeviceSynchronize();
  check_error();
}

template <class T>
void HIPStream<T>::read_arrays(std::vector<T>& a, std::vector<T>& b, std::vector<T>& c)
{
  // Copy device memory to host
  hipMemcpy(a.data(), d_a, a.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(b.data(), d_b, b.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
  hipMemcpy(c.data(), d_c, c.size()*sizeof(T), hipMemcpyDeviceToHost);
  check_error();
}

template <>
__global__ void read_kernel<float>(const float * a, float * sum, unsigned int array_size)
{
  int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4; //vec4 float to assemble a dword_x4
  __shared__ float threadBlock_sum[TBSIZE];

  a += i;

  threadBlock_sum[hipThreadIdx_x] = 0.0f;
  const int stride = hipBlockDim_x * hipGridDim_x * 4;

  float v0 = a[0];
  float v1 = a[1];
  float v2 = a[2];
  float v3 = a[3];

  float v4 = v0+v1;
  float v5 = v2+v3;
  threadBlock_sum[hipThreadIdx_x] += v4+v5;

  //fake loop to cheat compiler not to remove pure read kernel
  int offset = hipBlockDim_x/2;
  for(; offset > hipBlockDim_x/2; offset /= 2) 
  {
    __syncthreads();
    if(hipThreadIdx_x < offset)
    {
        threadBlock_sum[hipThreadIdx_x] += threadBlock_sum[hipThreadIdx_x + offset];
    }
  }

  if(hipThreadIdx_x == 0)
    sum[hipBlockIdx_x] = threadBlock_sum[0];
}

template <>
__global__ void read_kernel<double>(const double * a, double * sum, unsigned int array_size)
{
  int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 2; //vec2 double to assemble a dword_x4
  __shared__ float threadBlock_sum[TBSIZE];

  a += i;

  double v0 = a[0];
  double v1 = a[1];

  double v4 = v0+v1;
  threadBlock_sum[hipThreadIdx_x] = v4;

  //fake loop to cheat compiler not to remove pure read kernel
  int offset = hipBlockDim_x/2;
  for(; offset > hipBlockDim_x/2; offset /= 2)
  {
    if(hipThreadIdx_x < offset)
    {
      threadBlock_sum[hipThreadIdx_x] += threadBlock_sum[hipThreadIdx_x + offset];
    }
  }

  if(hipThreadIdx_x == 0)
    sum[hipBlockIdx_x] = threadBlock_sum[0];
}

template <class T>
T HIPStream<T>::read()
{
//  int readNumBlocks = array_size/(4*4*TBSIZE/sizeof(T));
  int readNumBlocks = READ_NUM_BLOCKS;
#ifdef EXT_KERNEL_TIME
  hipExtLaunchKernelGGL(HIP_KERNEL_NAME(read_kernel<T>), dim3(readNumBlocks), dim3(TBSIZE), 0, 0, start_ev, stop_ev, 0, d_a, sum_a, array_size);
  hipEventSynchronize(stop_ev);
  hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
#else
//  hipLaunchKernelGGL(HIP_KERNEL_NAME(copy_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_a, d_c);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(read_kernel<T>), dim3(readNumBlocks), dim3(TBSIZE), 0, 0, d_a, sum_a, array_size);
  check_error();
  hipDeviceSynchronize();
  check_error();
#endif

  hipMemcpy(sums_a, sum_a, readNumBlocks*sizeof(T), hipMemcpyDeviceToHost);
  check_error();

  T sum = 0.0;
  for(int i = 0; i < readNumBlocks; ++i)
  {
    sum = sums_a[i];
  }

  return sum;
}

template <>
__global__ void write_kernel<float>(float * d)
{
  const int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * WIDTH; //vec4 float to assemble a dword_x4

  //address offset explicitly, to avoid redundant address calc ISA
  d += i;

  d[0] = hipThreadIdx_x;
  d[1] = hipThreadIdx_x;
  d[2] = hipThreadIdx_x;
  d[3] = hipThreadIdx_x;
}

template <>
__global__ void write_kernel<double>(double *d)
{
  const int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * WIDTH/2; //vec2 double to assemble a dword_x4
  d += i;

  d[0] = hipThreadIdx_x;
  d[1] = hipThreadIdx_x;
}

template <class T>
void HIPStream<T>::write()
{
#ifdef EXT_KERNEL_TIME
  hipExtLaunchKernelGGL(HIP_KERNEL_NAME(write_kernel<T>), dim3(array_size/(4*WIDTH*TBSIZE/sizeof(T))), dim3(TBSIZE), 0, 0, start_ev, stop_ev, 0, d_d);
  hipEventSynchronize(stop_ev);
  hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
#else
  hipLaunchKernelGGL(HIP_KERNEL_NAME(write_kernel<T>), dim3(array_size/(4*WIDTH*TBSIZE/sizeof(T))), dim3(TBSIZE), 0, 0, d_d);
  check_error();
  hipDeviceSynchronize();
  check_error();
#endif
}

template <>
__global__ void copy_kernel<float>(const float * a, float * c)
{
  const int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 4; //vec4 float to assemble a dword_x4

  //address offset explicitly, to avoid redundant address calc ISA
  a += i;
  c += i;

  //read data in dword_x4 order, with fixed array index
  float v0 = a[0];
  float v1 = a[1];
  float v2 = a[2];
  float v3 = a[3];

  c[0] = v0;
  c[1] = v1;
  c[2] = v2;
  c[3] = v3;
}

template <>
__global__ void copy_kernel<double>(const double *a, double *c)
{
  const int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 2; //vec2 double to assemble a dword_x4
  a += i;
  c += i;

  double v0 = a[0];
  double v1 = a[1];

  c[0] = v0;
  c[1] = v1;
}
#if 0
template <typename T>
__global__ void copy_kernel(const T * a, T * c)
{
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  c[i] = a[i];
}
#endif

template <class T>
void HIPStream<T>::copy()
{
#ifdef EXT_KERNEL_TIME
  hipExtLaunchKernelGGL(HIP_KERNEL_NAME(copy_kernel<T>), dim3(array_size/(4*4*TBSIZE/sizeof(T))), dim3(TBSIZE), 0, 0, start_ev, stop_ev, 0, d_a, d_c);
  hipEventSynchronize(stop_ev);
  hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
#else
//  hipLaunchKernelGGL(HIP_KERNEL_NAME(copy_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_a, d_c);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(copy_kernel<T>), dim3(array_size/(4*4*TBSIZE/sizeof(T))), dim3(TBSIZE), 0, 0, d_a, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
#endif
}

template <>
__global__ void mul_kernel<float>(float * b, const float * c)
{
  const float scalar = startScalar;
  const int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x)*4;

  b += i;
  c += i;

  float v0 = c[0];
  float v1 = c[1];
  float v2 = c[2];
  float v3 = c[3];

  b[0] = scalar*v0;
  b[1] = scalar*v1;
  b[2] = scalar*v2;
  b[3] = scalar*v3;
}

template <>
__global__ void mul_kernel<double>(double * b, const double * c)
{
  const double scalar = startScalar;
  const int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x)*2;

  b += i;
  c += i;

  double v0 = c[0];
  double v1 = c[1];

  b[0] = scalar*v0;
  b[1] = scalar*v1;
}
#if 0
template <typename T>
__global__ void mul_kernel(T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  b[i] = scalar * c[i];
}
#endif

template <class T>
void HIPStream<T>::mul()
{
#ifdef EXT_KERNEL_TIME
  hipExtLaunchKernelGGL(HIP_KERNEL_NAME(mul_kernel<T>), dim3(array_size/(4*4*TBSIZE/sizeof(T))), dim3(TBSIZE), 0, 0, start_ev, stop_ev, 0, d_b, d_c);
  hipEventSynchronize(stop_ev);
  hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
#else
//  hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_b, d_c);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(mul_kernel<T>), dim3(array_size/(4*4*TBSIZE/sizeof(T))), dim3(TBSIZE), 0, 0, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
#endif
}

template <>
__global__ void add_kernel<float>(const float * a, const float * b, float * c)
{
  const int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x)*4;

  a += i;
  b += i;
  c += i;

  float v0 = a[0];
  float v1 = a[1];
  float v2 = a[2];
  float v3 = a[3];

  float v4 = b[0];
  float v5 = b[1];
  float v6 = b[2];
  float v7 = b[3];
  
  c[0] = v0 + v4;
  c[1] = v1 + v5;
  c[2] = v2 + v6;
  c[3] = v3 + v7;
}

template <>
__global__ void add_kernel<double>(const double * a, const double * b, double * c)
{
  const int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x)*2;

  a += i;
  b += i;
  c += i;

  double v0 = a[0];
  double v1 = a[1];

  double v4 = b[0];
  double v5 = b[1];

  c[0] = v0 + v4;
  c[1] = v1 + v5;
}

#if 0
template <typename T>
__global__ void add_kernel(const T * a, const T * b, T * c)
{
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  c[i] = a[i] + b[i];
}
#endif

template <class T>
void HIPStream<T>::add()
{
#ifdef EXT_KERNEL_TIME
  hipExtLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel<T>), dim3(array_size/(4*4*TBSIZE/sizeof(T))), dim3(TBSIZE), 0, 0, start_ev, stop_ev, 0, d_a, d_b, d_c);
  hipEventSynchronize(stop_ev);
  hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
#else
//  hipLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_a, d_b, d_c);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(add_kernel<T>), dim3(array_size/(4*4*TBSIZE/sizeof(T))), dim3(TBSIZE), 0, 0, d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
#endif
}

template <>
__global__ void triad_kernel<float>(float * a, const float * b, const float * c)
{
  const float scalar = startScalar;
  const int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x)*4;

  a += i;
  b += i;
  c += i;

  float v0 = c[0];
  float v1 = c[1];
  float v2 = c[2];
  float v3 = c[3];

  float v4 = b[0];
  float v5 = b[1];
  float v6 = b[2];
  float v7 = b[3];

  a[0] = v4 + scalar*v0;
  a[1] = v5 + scalar*v1;
  a[2] = v6 + scalar*v2;
  a[3] = v7 + scalar*v3;
}

template <>
__global__ void triad_kernel<double>(double * a, const double * b, const double * c)
{
  const double scalar = startScalar;
  const int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x)*2;

  a += i;
  b += i;
  c += i;

  double v0 = c[0];
  double v1 = c[1];

  double v4 = b[0];
  double v5 = b[1];

  a[0] = v4 + scalar*v0;
  a[1] = v5 + scalar*v1;
}

#if 0
template <typename T>
__global__ void triad_kernel(T * a, const T * b, const T * c)
{
  const T scalar = startScalar;
  const int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  a[i] = b[i] + scalar * c[i];
}
#endif

template <class T>
void HIPStream<T>::triad()
{
#ifdef EXT_KERNEL_TIME
  hipExtLaunchKernelGGL(HIP_KERNEL_NAME(triad_kernel<T>), dim3(array_size/(4*4*TBSIZE/sizeof(T))), dim3(TBSIZE), 0, 0, start_ev, stop_ev, 0, d_a, d_b, d_c);
  hipEventSynchronize(stop_ev);
  hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
#else
//  hipLaunchKernelGGL(HIP_KERNEL_NAME(triad_kernel<T>), dim3(array_size/TBSIZE), dim3(TBSIZE), 0, 0, d_a, d_b, d_c);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(triad_kernel<T>), dim3(array_size/(4*4*TBSIZE/sizeof(T))), dim3(TBSIZE), 0, 0, d_a, d_b, d_c);
  check_error();
  hipDeviceSynchronize();
  check_error();
#endif
}

template <>
__global__ void dot_kernel<float>(const float * a, const float * b, float * sum, unsigned int array_size)
{
  __shared__ float tb_sum[TBSIZE];

  int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x)*4;
  const size_t local_i = hipThreadIdx_x;

  a += i;
  b += i;

  float v0, v1, v2, v3, v4, v5, v6, v7;
  float v8, v9, v10, v11, v12, v13, v14;

  tb_sum[local_i] = 0.0f;
  const int stride = hipBlockDim_x * hipGridDim_x * 4;
  for( ; i < array_size; i += stride, a += stride, b += stride ) 
  {
    v0 = a[0];
    v1 = a[1];
    v2 = a[2];
    v3 = a[3];
    v4 = b[0];
    v5 = b[1];
    v6 = b[2];
    v7 = b[3];

    v8  = v0 * v4;
    v9  = v1 * v5;
    v10 = v2 * v6;
    v11 = v3 * v7;
    v12 = v8  + v9;
    v13 = v10 + v11;
    v14 = v12 + v13;
    tb_sum[local_i] += v14;
  }

  for(int offset = hipBlockDim_x / 2; offset > 0; offset /= 2)
  {
    __syncthreads();
    if(local_i < offset)
    {
      tb_sum[local_i] += tb_sum[local_i+offset];
    }
  }

  if (local_i == 0)
    sum[hipBlockIdx_x] = tb_sum[local_i];
}

template <>
__global__ void dot_kernel<double>(const double * a, const double * b, double * sum, unsigned int array_size)
{
  __shared__ double tb_sum[TBSIZE];

  int i = (hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x) * 2;
  const size_t local_i = hipThreadIdx_x;

  a += i;
  b += i;

  double v0, v1, v2, v3, v4, v5, v6, v7;
  double v8, v9, v10, v11, v12, v13, v14;

  tb_sum[local_i] = 0.0;
  const int stride = hipBlockDim_x * hipGridDim_x * 2;
  for( ; i < array_size; i += stride, a += stride, b += stride ) 
  {
    v0 = a[0];
    v1 = a[1];

    v4 = b[0];
    v5 = b[1];

    v8 = v0 * v4;
    v9 = v1 * v5;

    tb_sum[local_i] += v8+v9;
  }

  for(int offset = hipBlockDim_x / 2; offset > 0; offset /= 2)
  {
    __syncthreads();
    if(local_i < offset)
    {
      tb_sum[local_i] += tb_sum[local_i+offset];
    }
  }

  if (local_i == 0)
    sum[hipBlockIdx_x] = tb_sum[local_i];
}

#if 0
template <class T>
__global__ void dot_kernel(const T * a, const T * b, T * sum, unsigned int array_size)
{
  __shared__ T tb_sum[TBSIZE];

  int i = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  const size_t local_i = hipThreadIdx_x;

  tb_sum[local_i] = 0.0;
  T v0 = 0.0;
  for (; i < array_size; i += hipBlockDim_x*hipGridDim_x)
//    tb_sum[local_i] += a[i] * b[i];
    v0 += a[i] * b[i];
/*
  for (int offset = hipBlockDim_x / 2; offset > 0; offset /= 2)
  {
    __syncthreads();
    if (local_i < offset)
    {
      tb_sum[local_i] += tb_sum[local_i+offset];
    }
  }

  if (local_i == 0)
    sum[hipBlockIdx_x] = tb_sum[local_i];
*/
}
#endif

template <class T>
T HIPStream<T>::dot()
{
#ifdef EXT_KERNEL_TIME
  hipExtLaunchKernelGGL(HIP_KERNEL_NAME(dot_kernel<T>), dim3(DOT_NUM_BLOCKS), dim3(TBSIZE), 0, 0, start_ev, stop_ev, 0, d_a, d_b, d_sum, array_size);
  hipEventSynchronize(stop_ev);
  hipEventElapsedTime(&kernel_time, start_ev, stop_ev);
#else
//  hipLaunchKernelGGL(HIP_KERNEL_NAME(dot_kernel<T>), dim3(DOT_NUM_BLOCKS), dim3(TBSIZE), 0, 0, d_a, d_b, d_sum, array_size);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(dot_kernel<T>), dim3(DOT_NUM_BLOCKS), dim3(TBSIZE), 0, 0, d_a, d_b, d_sum, array_size);
  check_error();
#endif

  hipMemcpy(sums, d_sum, DOT_NUM_BLOCKS*sizeof(T), hipMemcpyDeviceToHost);
  check_error();

  T sum = 0.0;
  for (int i = 0; i < DOT_NUM_BLOCKS; i++)
    sum += sums[i];

  return sum;
}

void listDevices(void)
{
  // Get number of devices
  int count;
  hipGetDeviceCount(&count);
  check_error();

  // Print device names
  if (count == 0)
  {
    std::cerr << "No devices found." << std::endl;
  }
  else
  {
    std::cout << std::endl;
    std::cout << "Devices:" << std::endl;
    for (int i = 0; i < count; i++)
    {
      std::cout << i << ": " << getDeviceName(i) << std::endl;
    }
    std::cout << std::endl;
  }
}


std::string getDeviceName(const int device)
{
  hipDeviceProp_t props;
  hipGetDeviceProperties(&props, device);
  check_error();
  return std::string(props.name);
}


std::string getDeviceDriver(const int device)
{
  hipSetDevice(device);
  check_error();
  int driver;
  hipDriverGetVersion(&driver);
  check_error();
  return std::to_string(driver);
}

template class HIPStream<float>;
template class HIPStream<double>;
