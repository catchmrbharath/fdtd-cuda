from numpy import *

from matplotlib.pyplot import *
EPS = 8.8541878176e-12
MU = 1.2566370614e-6
xdim = 1024
ydim = 1024
dx = 20e-6 / 400
ta = ones((xdim, ydim)) * EPS
ta2 = ones((xdim, ydim)) * MU
ta3 = zeros((xdim, ydim))

ny1 = round((ydim / 2) - (0.25e-6) / dx)
ny2 = round((ydim / 2) + (0.25e-6) / dx)


ny3 = round((ydim / 2) - (1.25e-6) / dx)
ny4 = round((ydim / 2) - (0.75e-6) / dx)

ny5 = round((ydim / 2) + (0.75e-6) / dx)
ny6 = round((ydim / 2) + (1.25e-6) / dx)

ta[:, ny1:ny2] = 2.25 * EPS
ta[:,ny3:ny4] = 2.25 * EPS
ta[:, ny5:ny6] = 2.25 * EPS

savetxt('eps_new.csv', ta,  delimiter = " ")
savetxt('mus_new.csv', ta2,  delimiter = " ")
savetxt('sigma_new.csv', ta2,  delimiter = " ")
