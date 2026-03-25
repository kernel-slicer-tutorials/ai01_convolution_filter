#pragma once
#include <vector>
#include <cstdint>
#include <chrono>
#include <memory>
#include <algorithm>
#include <cmath>

#include "LiteMath.h"

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;

class Convolution2D
{
public:
  Convolution2D(){}

  // Setup kernel functions (fill m_kernel, m_kw, m_kh)
  void SetupBlurKernel(int radius);
  void SetupSharpenKernel();
  void SetupEdgeDetectKernel();
  void SetupEmbossKernel();
  void SetupGaussianBlurKernel(int radius, float sigma);

  virtual void Apply(int w, int h,
                     const float* inData  [[size("w*h*4")]],
                     float*       outData [[size("w*h*4")]]);

  virtual void CommitDeviceData(){}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]){ a_out[0] = m_time; }

protected:
  virtual void kernel2D_convolve(int w, int h,
                                 const float* inData,
                                 float*       outData);

  // convolution kernel stored as flat array [kh * kw]
  std::vector<float> m_kernel;
  int   m_kw = 1;
  int   m_kh = 1;
  float m_time = 0.0f;
};
