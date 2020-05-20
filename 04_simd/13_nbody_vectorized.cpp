#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for (int i = 0; i < N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  // Load x and y vectors
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);

  // Init 0 vector and 1 vector for masks
  __m256 zeros = _mm256_set1_ps(0);
  __m256 ones = _mm256_set1_ps(1);

  // Create stack arrays for reduction
  float fx_stack[N][N], fy_stack[N][N];

  for (int i = 0; i < N; i++) {
    // Load m vector of bodies
    // Reload every iteration, as the ith entry needs to be set to 0
    __m256 mvec = _mm256_load_ps(m);

    // Create vector with only x[i] and y[i] for subtraction
    __m256 x_i = _mm256_set1_ps(x[i]);
    __m256 y_i = _mm256_set1_ps(y[i]);

    // Compute r with Pythagoras
    __m256 rx = _mm256_sub_ps(x_i, xvec);
    __m256 ry = _mm256_sub_ps(y_i, yvec);
    __m256 rx_sq = _mm256_mul_ps(rx, rx);
    __m256 ry_sq = _mm256_mul_ps(ry, ry);
    __m256 r_sq = _mm256_add_ps(rx_sq, ry_sq);
    __m256 r = _mm256_sqrt_ps(r_sq);

    // Set ith entry of the m vector to 0
    __m256 r_mask = _mm256_cmp_ps(r, zeros, _CMP_GT_OQ);
    mvec = _mm256_blendv_ps(zeros, mvec, r_mask);

    // Set the ith entry of the r_cube vector to 1 to prevent div by 0
    __m256 r_cube = _mm256_mul_ps(r, r_sq);
    r_cube = _mm256_blendv_ps(ones, r_cube, r_mask);

    // Do final calculations to get vector to be reduced
    __m256 fxvec = _mm256_mul_ps(rx, mvec);
    __m256 fyvec = _mm256_mul_ps(ry, mvec);
    fxvec = _mm256_div_ps(fxvec, r_cube);
    fyvec = _mm256_div_ps(fyvec, r_cube);

    // Put vector to reduce in memory stack
    _mm256_store_ps(fx_stack[i], fxvec);
    _mm256_store_ps(fy_stack[i], fyvec);
  }

  // Perform reduction to obtain final fx and fy values
  // The compiler will vectorize this for-loop
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      fx[i] -= fx_stack[i][j];
      fy[i] -= fy_stack[i][j];
    }
  }

  for (int i = 0; i < N; i++) {
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
