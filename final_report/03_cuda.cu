#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

const char *outf = "03_cuda.out";

const int N_THREADS = 1024; // threads per block

__global__ void build_up_b(int nx, int ny, float *u, float *v, float *b,
                           float dt, float dx, float dy, float rho) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx)
    return;
  if (i % nx == nx - 1)
    return;
  if (i % nx == 0)
    return;
  if (i >= nx * (ny - 1))
    return;

  b[i] = rho * (1 / dt *
                    ((u[i + 1] - u[i - 1]) / (2. * dx) +
                     (v[i + nx] - v[i - nx]) / (2. * dy)) -
                pow((u[i + 1] - u[i - 1]) / (2. * dx), 2) -
                2. * ((u[i + nx] - u[i - nx]) / (2. * dy) *
                      (v[i + 1] - v[i - 1]) / (2. * dx)) -
                pow((v[i + nx] - v[i - nx]) / (2. * dy), 2));
}

__global__ void pressure_poisson(int nx, int ny, float *p, float *pn, float *b,
                                 float dx, float dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if ((i >= nx) && (i % nx != nx - 1) && (i % nx != 0) && (i < nx * (ny - 1))) {
    p[i] = ((pn[i + 1] + pn[i - 1]) * dy * dy +
            (pn[i + nx] + pn[i - nx]) * dx * dx) /
               (2 * (dx * dx + dy * dy)) -
           dx * dx * dy * dy / (2 * (dx * dx + dy * dy)) * b[i];
  }
}

__global__ void pressure_poisson_boundary(int nx, int ny, float *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nx * ny)
    return;
  if (i % nx == nx - 1)
    p[i] = p[i - 1];
  else if (i < nx)
    p[i] = p[i + nx];
  else if (i % nx == 0)
    p[i] = p[i + 1];
  else if (i >= nx * (ny - 1))
    p[i] = 0;
}

__global__ void compute_uv(int nx, int ny, float *u, float *v, float *un,
                           float *vn, float *p, float dt, float dx, float dy,
                           float rho, float nu) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx * ny) {
    if ((i >= nx) && (i % nx != nx - 1) && (i % nx != 0) &&
        (i < nx * (ny - 1))) {
      u[i] = un[i] - un[i] * dt / dx * (un[i] - un[i - 1]) -
             vn[i] * dt / dy * (un[i] - un[i - nx]) -
             dt / (2. * rho * dx) * (p[i + 1] - p[i - 1]) +
             nu * (dt / (dx * dx) * (un[i + 1] - 2. * un[i] + un[i - 1]) +
                   dt / (dy * dy) * (un[i + nx] - 2. * un[i] + un[i - nx]));
    }
  } else {
    i -= nx * ny;
    if ((i >= nx) && (i % nx != nx - 1) && (i % nx != 0) &&
        (i < nx * (ny - 1))) {
      v[i] = vn[i] - un[i] * dt / dx * (vn[i] - vn[i - 1]) -
             vn[i] * dt / dy * (vn[i] - vn[i - nx]) -
             dt / (2. * rho * dy) * (p[i + nx] - p[i - nx]) +
             nu * (dt / (dx * dx) * (vn[i + 1] - 2. * vn[i] + vn[i - 1]) +
                   dt / (dy * dy) * (vn[i + nx] - 2. * vn[i] + vn[i - nx]));
    }
  }
}

__global__ void compute_uv_boundary(int nx, int ny, float *u, float *v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nx * ny) {
    if ((i % nx == nx - 1) && (i < nx * (ny - 1)))
      u[i] = 0;
    else if (i < nx)
      u[i] = 0;
    else if ((i % nx == 0) && (i < nx * (ny - 1)))
      u[i] = 0;
    else if (i >= nx * (ny - 1))
      u[i] = 1;
  } else {
    i -= nx * ny;
    if (i >= nx * ny)
      return;
    if (i % nx == nx - 1)
      v[i] = 0;
    else if (i < nx)
      v[i] = 0;
    else if (i % nx == 0)
      v[i] = 0;
    else if (i >= nx * (ny - 1))
      v[i] = 0;
  }
}

__device__ __managed__ float u_sum;
__device__ __managed__ float u_diff;

__device__ float warp_sum(float a) {
  for (int offset = 16; offset > 0; offset >>= 1)
    a += __shfl_down_sync(0xffffffff, a, offset);
  return a;
}

__global__ void l1_diff(int n, float *a, float *an, float &diff) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;

  float b = warp_sum(abs(a[i] - an[i]));
  if ((threadIdx.x & 31) == 0)
    atomicAdd(&diff, b);
}

__global__ void l1_sum(int n, float *a, float &sum) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float b = warp_sum(abs(a[i]));
  if ((threadIdx.x & 31) == 0)
    atomicAdd(&sum, b);
}

int cavity_flow(float eps, int nx, int ny, float *u, float *un, float *v,
                float *vn, float *p, float *pn, float *b, float dt, float dx,
                float dy, float rho, float nu) {

  const int grid_size = nx * ny;
  const int size = grid_size * sizeof(float);
  int n_blocks = (grid_size + N_THREADS - 1) / N_THREADS;

  int nt = 0;
  u_diff = 1000;
  for (; u_diff > eps; nt++) {
    cudaMemcpy(un, u, size, cudaMemcpyDefault);
    cudaMemcpy(vn, v, size, cudaMemcpyDefault);

    build_up_b<<<n_blocks, N_THREADS>>>(nx, ny, u, v, b, dt, dx, dy, rho);
    for (int nit = 0; nit < 50; nit++) {
      cudaMemcpy(pn, p, size, cudaMemcpyDefault);
      pressure_poisson<<<n_blocks, N_THREADS>>>(nx, ny, p, pn, b, dx, dy);
      pressure_poisson_boundary<<<n_blocks, N_THREADS>>>(nx, ny, p);
    }

    compute_uv<<<n_blocks * 2, N_THREADS>>>(nx, ny, u, v, un, vn, p, dt, dx, dy,
                                            rho, nu);
    compute_uv_boundary<<<n_blocks * 2, N_THREADS>>>(nx, ny, u, v);

    // diff on u to check for convergence
    u_diff = 0;
    u_sum = 0;
    l1_diff<<<n_blocks, N_THREADS>>>(grid_size, u, un, u_diff);
    l1_sum<<<n_blocks, N_THREADS>>>(grid_size, u, u_sum);
    cudaDeviceSynchronize();
    u_diff /= u_sum;
  }
  return nt;
}

int main() {
  const int nx = 41;
  const int ny = 41;
  const int grid_size = ny * nx;

  const float dx = 2. / (nx - 1.);
  const float dy = 2. / (ny - 1.);

  const float rho = 1.;
  const float nu = .1;
  const float dt = .001;

  float *u, *v, *p, *un, *vn, *pn, *b;
  const int size = grid_size * sizeof(float);
  cudaMallocManaged(&u, size);
  cudaMallocManaged(&v, size);
  cudaMallocManaged(&p, size);
  cudaMallocManaged(&un, size);
  cudaMallocManaged(&vn, size);
  cudaMallocManaged(&pn, size);
  cudaMallocManaged(&b, size);
  cudaMemset(u, 0, size);
  cudaMemset(v, 0, size);
  cudaMemset(p, 0, size);
  cudaMemset(un, 0, size);
  cudaMemset(vn, 0, size);
  cudaMemset(pn, 0, size);
  cudaMemset(b, 0, size);

  float eps = .0000001;
  auto start_time = chrono::steady_clock::now();
  int nt =
      cavity_flow(eps, nx, ny, u, un, v, vn, p, pn, b, dt, dx, dy, rho, nu);
  auto end_time = chrono::steady_clock::now();
  double time = chrono::duration<double>(end_time - start_time).count();
  printf("Steps: %d\n", nt);
  printf("Elapsed time: %lf s.\n", time);

  float u_sum = 0, v_sum = 0, p_sum = 0;
  for (int i = 0; i < grid_size; i++) {
    u_sum += abs(u[i]);
    v_sum += abs(v[i]);
    p_sum += abs(p[i]);
  }
  printf("Sum(|u|)=%f\n", u_sum);
  printf("Sum(|v|)=%f\n", v_sum);
  printf("Sum(|p|)=%f\n", p_sum);

  ofstream f(outf);
  if (f.is_open()) {
    f << ny << " ";
    f << nx << "\n";

    for (int i = 0; i < grid_size; i++)
      f << u[i] << " ";
    f << "\n";

    for (int i = 0; i < grid_size; i++)
      f << v[i] << " ";
    f << "\n";

    for (int i = 0; i < grid_size; i++)
      f << p[i] << " ";

    f.close();
  }

  cudaFree(u);
  cudaFree(v);
  cudaFree(p);
  cudaFree(un);
  cudaFree(vn);
  cudaFree(pn);
  cudaFree(b);

  return 0;
}