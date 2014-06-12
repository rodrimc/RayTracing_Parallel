RayTracing
==========

This is a Ray Tracing implementation in C++.

This code implements rays' intersection with spheres and plane.
There is also reflection.

It was implemented and tested on ubuntu 14.04, compiled using
g++ 4.8.2 and depends on SDL2 to render the scene. There is
a minimalist makefile.

To compile just clone the repository and type: make. You
need the SDL2 to compile it withou problems.

The executable will be created in a folder "build". To run
the program, just type ./RayTracingT3, passing as argument the
width, height and the fov (field of view) desired -- this last
argumnet is optional (default value is 60º).

For example, the following image was generated typing the
following command after build the source:

./RayTracingT3 800 600 60

![RayTracingT3](https://raw.githubusercontent.com/rodrimc/RayTracing/master/RayTracerT3.png)

Fell free to contribute with this code.
