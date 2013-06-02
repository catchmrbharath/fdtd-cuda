from __future__ import division
import imp
import sys
from numpy import zeros, ones, log, savetxt, loadtxt
from matplotlib.pyplot import *
load_source = imp.load_source;
EPS = 8.8541878176e-12
MU = 1.2566370614e-6
LIGHTSPEED = 299792458

def make_pml(sigma, eps, mu,  pml_size, delta):
    xdim = sigma.shape[0]
    ydim = sigma.shape[1]
    sigma_x = zeros((xdim + 2 * pml_size, ydim + 2 * pml_size), dtype='float')
    sigma_y = zeros((xdim + 2 * pml_size, ydim + 2 * pml_size), dtype='float')
    sigma_star_x = zeros((xdim + 2 * pml_size, ydim + 2 * pml_size), dtype='float')
    sigma_star_y = zeros((xdim + 2 * pml_size, ydim + 2 * pml_size), dtype='float')
    eps_new = ones((xdim + 2 * pml_size, ydim + 2 * pml_size), dtype='float') * EPS
    mu_new = ones((xdim + 2 * pml_size, ydim + 2 * pml_size), dtype='float') * MU

    sigma_x[pml_size:-pml_size, pml_size:-pml_size] = sigma
    sigma_y[pml_size:-pml_size, pml_size:-pml_size] = sigma
    eps_new[pml_size:-pml_size, pml_size:-pml_size] = eps
    mu_new[pml_size:-pml_size, pml_size:-pml_size] = mu
    for i in range(pml_size):
        eps_new[i, pml_size:-pml_size] = eps[0, :]
        eps_new[xdim + i, pml_size:-pml_size] = eps[xdim - 1, :]
        eps_new[pml_size:-pml_size, i] = eps[:, 0]
        eps_new[pml_size:-pml_size, ydim + i] = eps[:, ydim - 1]
        mu_new[i, pml_size:-pml_size] = mu[0, :]
        mu_new[xdim + i, pml_size:-pml_size] = mu[xdim - 1, :]
        mu_new[pml_size:-pml_size, i] = mu[:, 0]
        mu_new[pml_size:-pml_size, ydim+ i] = mu[:, ydim - 1]

    sigma_max = -log(1e-6) * (3.0 + 1) * EPS * LIGHTSPEED / (2.0 * pml_size)
    sigma_max = sigma_max / delta
    for i in range(pml_size):
        sigma_x[i, :] = sigma_max * ((pml_size - i) / pml_size)**4
        sigma_y[:, i] = sigma_max * ((pml_size - i) / pml_size)**4

    for i in range(pml_size, 0, -1):
        sigma_x[xdim + 2 * pml_size - i, :] = sigma_max * ((pml_size - i) / pml_size)**4
        sigma_y[:, ydim + 2 * pml_size - i] = sigma_max * ((pml_size - i) / pml_size)**4

    sigma_star_y[:, :pml_size] = sigma_y[:, :pml_size] * MU / EPS
    sigma_star_y[:, ydim:ydim + 2 * pml_size] = sigma_y[:, ydim: ydim + 2 * pml_size] * MU / EPS

    sigma_star_x[:pml_size, :] = sigma_x[:pml_size, :] * MU / EPS
    sigma_star_x[xdim:xdim + 2 * pml_size, :] = sigma_x[xdim:xdim + 2 * pml_size, :] * MU / EPS
    imshow(eps_new)
    show()
    imshow(mu_new)
    show()

    imshow(sigma_x)
    show()

    imshow(sigma_y)
    show()

    imshow(sigma_star_x)
    show()

    imshow(sigma_star_y)
    show()
    savetxt('../eps_changed.csv', eps_new, delimiter =" ")
    savetxt('../mu_changed.csv', mu_new, delimiter =" ")
    savetxt('../sigma_x_changed.csv', sigma_x, delimiter =" ")
    savetxt('../sigma_y_changed.csv', sigma_y, delimiter =" ")
    savetxt('../sigma_star_x_changed.csv', sigma_star_x, delimiter =" ")
    savetxt('../sigma_star_y_changed.csv', sigma_star_y, delimiter =" ")


if len(sys.argv) == 2:
    mod = load_source('config', sys.argv[1]);
    print mod.type
    print mod.dx
    if mod.type == 1:
        print mod.xdim + 2 * mod.pml_width
        print mod.ydim + 2 * mod.pml_width
    else:
        print mod.xdim
        print mod.ydim


    if mod.type == 0:
        print mod.sigma
        print mod.eps
        print mod.mu

    elif mod.type == 1:
        sigma = loadtxt(mod.sigma)
        eps = loadtxt(mod.eps)
        mu = loadtxt(mod.mu)
        make_pml(sigma, eps, mu,  mod.pml_width, mod.dx)
        print 'eps_changed.csv'
        print 'mu_changed.csv'
        print 'sigma_x_changed.csv'
        print 'sigma_y_changed.csv'
        print 'sigma_star_x_changed.csv'
        print 'sigma_star_y_changed.csv'
        for sources in mod.sources:
            print sources[0], sources[1], sources[2], sources[3], sources[4]
