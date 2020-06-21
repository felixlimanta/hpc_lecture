#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

typedef vector<vector<double>> matrix;

void build_up_b(int nx, int ny, matrix &b, double rho, double dt, matrix &u,
                matrix &v, double dx, double dy) {

  for (int i = 1; i < ny - 1; i++) {
    for (int j = 1; j < nx - 1; j++) {
      b[i][j] = rho * (1 / dt *
                           ((u[i][j + 1] - u[i][j - 1]) / (2. * dx) +
                            (v[i + 1][j] - v[i - 1][j]) / (2. * dy)) -
                       pow((u[i][j + 1] - u[i][j - 1]) / (2. * dx), 2) -
                       2. * ((u[i + 1][j] - u[i - 1][j]) / (2. * dy) *
                             (v[i][j + 1] - v[i][j - 1]) / (2. * dx)) -
                       pow((v[i + 1][j] - v[i - 1][j]) / (2. * dy), 2));
    }
  }
}

int nit = 50;
void pressure_poisson(int nx, int ny, matrix &p, double dx, double dy,
                      matrix &b) {

  matrix pn(ny, vector<double>(nx));
  for (int q = 0; q < nit; q++) {
    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        pn[i][j] = p[i][j];
      }
    }
    for (int i = 1; i < ny - 1; i++) {
      for (int j = 1; j < nx - 1; j++) {
        p[i][j] = ((pn[i][j + 1] + pn[i][j - 1]) * dy * dy +
                   (pn[i + 1][j] + pn[i - 1][j]) * dx * dx) /
                      (2 * (dx * dx + dy * dy)) -
                  dx * dx * dy * dy / (2 * (dx * dx + dy * dy)) * b[i][j];
      }
    }

    for (int i = 0; i < ny; i++) {
      p[i][nx - 1] = p[i][nx - 2]; // dp/dx = 0 at x = 2
      p[i][0] = p[i][1];           // dp/dy = 0 at y = 0
    }
    for (int i = 0; i < nx; i++) {
      p[0][i] = p[1][i]; // dp/dx = 0 at x = 0
      p[ny - 1][i] = 0;  // p = 0 at y = 2
    }
  }
}

int cavity_flow(double eps, int nx, int ny, matrix &u, matrix &v, double dt,
                double dx, double dy, matrix &p, double rho, double nu) {

  matrix un(ny, vector<double>(nx));
  matrix vn(ny, vector<double>(nx));
  matrix b(ny, vector<double>(nx));

  int nt = 0;
  double du = 1000;
  for (; du > eps; nt++) {
    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        un[i][j] = u[i][j];
        vn[i][j] = v[i][j];
      }
    }

    build_up_b(nx, ny, b, rho, dt, u, v, dx, dy);
    pressure_poisson(nx, ny, p, dx, dy, b);

    for (int i = 1; i < ny - 1; i++) {
      for (int j = 1; j < nx - 1; j++) {
        u[i][j] = un[i][j] - un[i][j] * dt / dx * (un[i][j] - un[i][j - 1]) -
                  vn[i][j] * dt / dy * (un[i][j] - un[i - 1][j]) -
                  dt / (2. * rho * dx) * (p[i][j + 1] - p[i][j - 1]) +
                  nu * (dt / (dx * dx) *
                            (un[i][j + 1] - 2. * un[i][j] + un[i][j - 1]) +
                        dt / (dy * dy) *
                            (un[i + 1][j] - 2. * un[i][j] + un[i - 1][j]));
      }
    }

    for (int i = 1; i < ny - 1; i++) {
      for (int j = 1; j < nx - 1; j++) {
        v[i][j] = vn[i][j] - un[i][j] * dt / dx * (vn[i][j] - vn[i][j - 1]) -
                  vn[i][j] * dt / dy * (vn[i][j] - vn[i - 1][j]) -
                  dt / (2. * rho * dy) * (p[i + 1][j] - p[i - 1][j]) +
                  nu * (dt / (dx * dx) *
                            (vn[i][j + 1] - 2. * vn[i][j] + vn[i][j - 1]) +
                        dt / (dy * dy) *
                            (vn[i + 1][j] - 2. * vn[i][j] + vn[i - 1][j]));
      }
    }

    for (int i = 0; i < ny; i++) {
      u[i][0] = 0;
      u[i][nx - 1] = 0;
      v[i][0] = 0;
      v[i][nx - 1] = 0;
    }

    for (int i = 0; i < nx; i++) {
      u[0][i] = 0;
      u[ny - 1][i] = 1; // set velocity on cavity lid equal to 1
      v[0][i] = 0;
      v[ny - 1][i] = 0;
    }

    // diff on u to check for convergence
    du = 0;
    double u_sum = 0;
    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        du += abs(u[i][j] - un[i][j]);
        u_sum += abs(u[i][j]);
      }
    }
    du /= u_sum;
  }

  return nt;
}

int main() {
  int nx = 41;
  int ny = 41;
  double dx = 2. / (nx - 1.);
  double dy = 2. / (ny - 1.);

  double rho = 1.;
  double nu = .1;
  double dt = .001;

  matrix u(ny, vector<double>(nx));
  matrix v(ny, vector<double>(nx));
  matrix p(ny, vector<double>(nx));

  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++) {
      u[i][j] = 0;
      v[i][j] = 0;
      p[i][j] = 0;
    }
  }

  double eps = .0000001;
  auto start_time = chrono::steady_clock::now();
  int nt = cavity_flow(eps, nx, ny, u, v, dt, dx, dy, p, rho, nu);
  auto end_time = chrono::steady_clock::now();
  double time = chrono::duration<double>(end_time - start_time).count();
  printf("Steps: %d\n", nt);
  printf("Elapsed time: %lf s.\n", time);

  double u_sum = 0, v_sum = 0, p_sum = 0;
  for (int i = 0; i < ny; i++) {
    for (int j = 0; j < nx; j++) {
      u_sum += abs(u[i][j]);
      v_sum += abs(v[i][j]);
      p_sum += abs(p[i][j]);
    }
  }
  printf("Sum(|u|)=%f\n", u_sum);
  printf("Sum(|v|)=%f\n", v_sum);
  printf("Sum(|p|)=%f\n", p_sum);

  ofstream f("02_navier_stokes.out");
  if (f.is_open()) {
    f << ny << " " << nx << "\n";

    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        f << u[i][j] << " ";
      }
    }
    f << "\n";

    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        f << v[i][j] << " ";
      }
    }
    f << "\n";

    for (int i = 0; i < ny; i++) {
      for (int j = 0; j < nx; j++) {
        f << p[i][j] << " ";
      }
    }

    f.close();
  }

  return 0;
}