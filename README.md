RayTracing Parallel
-----------------

This is a Ray Tracing implementation in C++ using OpenMP and Cuda.

This code implements rays' intersection with spheres and plane.
There is also reflection.

It was implemented and tested on ubuntu 14.04, compiled using
g++ 4.8.2. There is a minimalist makefile that builds three executables:
RayTracing RayTracing_openmp and RayTracing_cuda.

Run
-----------------
The executables will be created in the folder "build". To run
the program, just type ./RayTracing_<version>, passing as argument the
width, height and the fov (field of view) desired -- this last
argumnet is optional (default value is 60º).

Example
-----------------
The following image was generated typing the
following command after build the source:

./RayTracing_openmp 800 600 60

![RayTracing_Parallel](https://raw.githubusercontent.com/rodrimc/RayTracing_Parallel/master/out.ppm)

Fell free to contribute.
