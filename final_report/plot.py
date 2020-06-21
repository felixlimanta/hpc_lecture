import argparse
import numpy

from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots Navier-Stokes results from C++ calculations')
    parser.add_argument('inf', metavar='in', type=str, help='Input filename')
    parser.add_argument('outf', metavar='in', type=str, help='Input filename')

    args = parser.parse_args()

    with open(args.inf) as inf:
        line = inf.readline()
        ny, nx = line.split()
        ny = int(ny)
        nx = int(nx)

        grid_size = ny * nx
        u = numpy.zeros(grid_size)
        v = numpy.zeros(grid_size)
        p = numpy.zeros(grid_size)

        line = inf.readline()
        c = 0
        for i in line.split():
            u[c] = float(i)
            c += 1

        line = inf.readline()
        c = 0
        for i in line.split():
            v[c] = float(i)
            c += 1

        line = inf.readline()
        c = 0
        for i in line.split():
            p[c] = float(i)
            c += 1

    u = u.reshape((ny, nx))
    v = v.reshape((ny, nx))
    p = p.reshape((ny, nx))

    print("Figure exported as", args.outf)

    x = numpy.linspace(0, 2, nx)
    y = numpy.linspace(0, 2, ny)
    X, Y = numpy.meshgrid(x, y)
    fig = pyplot.figure(figsize=(11, 7), dpi=100)
    pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
    pyplot.colorbar()
    pyplot.contour(X, Y, p, cmap=cm.viridis)
    pyplot.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
    pyplot.xlabel('X')
    pyplot.ylabel('Y')
    pyplot.savefig(args.outf)
