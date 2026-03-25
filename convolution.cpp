#include "convolution.h"

// ---------------------------------------------------------------------------
// Kernel setup helpers
// ---------------------------------------------------------------------------

void Convolution2D::SetupBlurKernel(int radius)
{
  m_kw = 2 * radius + 1;
  m_kh = 2 * radius + 1;
  int sz = m_kw * m_kh;
  m_kernel.assign(sz, 1.0f / (float)sz);
}

void Convolution2D::SetupSharpenKernel()
{
  m_kw = 3; m_kh = 3;
  m_kernel = {
     0.0f, -1.0f,  0.0f,
    -1.0f,  5.0f, -1.0f,
     0.0f, -1.0f,  0.0f
  };
}

void Convolution2D::SetupEdgeDetectKernel()
{
  m_kw = 3; m_kh = 3;
  // Laplacian edge detection
  m_kernel = {
    -1.0f, -1.0f, -1.0f,
    -1.0f,  8.0f, -1.0f,
    -1.0f, -1.0f, -1.0f
  };
}

void Convolution2D::SetupEmbossKernel()
{
  m_kw = 3; m_kh = 3;
  m_kernel = {
    -2.0f, -1.0f,  0.0f,
    -1.0f,  1.0f,  1.0f,
     0.0f,  1.0f,  2.0f
  };
}

void Convolution2D::SetupGaussianBlurKernel(int radius, float sigma)
{
  m_kw = 2 * radius + 1;
  m_kh = 2 * radius + 1;
  m_kernel.resize(m_kw * m_kh);

  float sum = 0.0f;
  for (int ky = -radius; ky <= radius; ++ky)
  {
    for (int kx = -radius; kx <= radius; ++kx)
    {
      float val = std::exp(-(kx * kx + ky * ky) / (2.0f * sigma * sigma));
      int idx = (ky + radius) * m_kw + (kx + radius);
      m_kernel[idx] = val;
      sum += val;
    }
  }
  // normalize
  for (auto& v : m_kernel)
    v /= sum;
}

// ---------------------------------------------------------------------------
// kernel2D_convolve  –– the parallel kernel (2D loop)
// ---------------------------------------------------------------------------

void Convolution2D::kernel2D_convolve(int w, int h,
                                      const float* inData,
                                      float*       outData)
{

  for (int y = 0; y < h; y++)
  {
    for (int x = 0; x < w; x++)
    {
      const int halfKW = m_kw / 2;
      const int halfKH = m_kh / 2;
      float r = 0.0f, g = 0.0f, b = 0.0f, a = 0.0f;

      for (int ky = 0; ky < m_kh; ++ky)
      {
        for (int kx = 0; kx < m_kw; ++kx)
        {
          int sx = x + kx - halfKW;
          int sy = y + ky - halfKH;

          // clamp to border
          sx = std::max(0, std::min(sx, w - 1));
          sy = std::max(0, std::min(sy, h - 1));

          float kval = m_kernel[ky * m_kw + kx];
          int   pidx = 4 * (sy * w + sx);

          r += inData[pidx + 0] * kval;
          g += inData[pidx + 1] * kval;
          b += inData[pidx + 2] * kval;
          a += inData[pidx + 3] * kval;
        }
      }

      int oidx = 4 * (y * w + x);
      outData[oidx + 0] = r;
      outData[oidx + 1] = g;
      outData[oidx + 2] = b;
      outData[oidx + 3] = a;
    }
  }
}

// ---------------------------------------------------------------------------
// Control function Apply
// ---------------------------------------------------------------------------

void Convolution2D::Apply(int w, int h,
                          const float* inData,
                          float*       outData)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel2D_convolve(w, h, inData, outData);
  m_time = std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now() - before).count() / 1000.0f;
}
