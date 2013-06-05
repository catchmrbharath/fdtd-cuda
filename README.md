FDTD Algorithm On a GPU                        {#mainpage}
=======================

Introduction
------------

This is an implementation of the FDTD algorithm in 2 dimensions on a GPU.
It supports all TM modes and also has support for an PML(split field) and 
has implementation for dispersive materials.

Requirements
-------------

* Cuda
* OpenGL (Do a sudo apt-get install freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libgl1-mesa-glx libglu1-mesa libglu1-mesa-dev)
to install the requirements.
* hdf5 
* h5utils
* Python

The walkthrough a problem is [here](https://github.com/catchmrbharath/fdtd-cuda/wiki/coupler).
