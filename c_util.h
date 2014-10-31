
/*
 * util_cuda.h
 *
 *  Created on: Oct 31, 2014
 *      Author: Rodrigo Costa
 *      email:  rodrigocosta@telemidia.puc-rio.br
 */

#ifndef UTILCUDA_H_
#define UTILCUDA_H_

#include "c_IShape.h"
#include "c_Plane.h"
#include "c_Sphere.h"
#include "c_Light.h"
#include "c_Color.h"

#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

typedef Color Image;

__global__ void k_trace (Image *d_image, int width, int height, 
    float aspect_ratio, float tanFov, int depth)
{
  int outerOffset = (blockIdx.x * gridDim.y + blockIdx.y) 
    * (blockDim.x * blockDim.y);
  int innerOffset = threadIdx.x * blockDim.y + threadIdx.y;

  int final_offset = outerOffset + innerOffset;

  if (final_offset < width * height)
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float yu = (1 - 2 * ((y + 0.5) * 1 / height)) * tanFov;
    float xu = (2 * ((x + 0.5) * 1 / float (width)) - 1) * tanFov
      * aspect_ratio;
	  Point origin (0.0f, 5.0f, 20.0f);
    
    Ray ray (origin, Vector3D (xu, yu, -1));

    d_image[final_offset] = Color (1.0f, 0.0f, 0.0f);
  }
}


void c_initScene (thrust::host_vector<IShape*> &shapes,
    thrust::host_vector<Light*> &lights)
{
  Plane *mirror = new Plane (Point(0.0f, 0.0, -15.0f), Vector3D (0.0f, 0.0f, 1.0f), 
      Color(0.2f, 0.2f, 0.2f), 1.0f, 0.2f, 0.2f, false);

  Plane *floor = new Plane(Point(0.0f, -2.5f, 0.0f), Vector3D(0.0f, 1.0f, 0.0f),
      Color(1.0f, 1.0f, 1.0f), 0.5f, 0.3f, 0.3f);

  Sphere *sphere0 = new Sphere(Point(-5.0f, 1.0f, 0.0f), Color(0.5, 0.1, 0.1),
      0.5f, 0.5f, 1.0f, 0.6f);

  Sphere *sphere1 = new Sphere(Point(3.0f, 0.0f, 0.0f), Color(1.0, 0.1, 0.1),
      1.5f, 0.8f, 1.0f, 0.2f);
  Sphere *sphere2 = new Sphere(Point(-3.0f, 0.0f, 0.0f), Color(0.1, 1.0, 0.1),
      1.5f, 0.5f, 1.0f, 0.7f);
  Sphere *sphere3 = new Sphere(Point(0.0f, 0.0f, -4.0f), Color(0.5, 0.5, 0.5),
      1.5f, 0.5f, 1.0f, 0.7f);

  Sphere *sphere4 = new Sphere(Point(10.0f, 5.0f, -4.0f), Color(1.0, 0.3, 0.8),
      1.5f, 0.8f, 1.0f, 0.4f);

  Sphere *sphere5 = new Sphere(Point(8.0f, 0.0f, 4.0f), Color(0.5, 0.5, 1.0),
      1.5f, 0.3f, 1.0f, 0.7f);

  Sphere *sphere6 = new Sphere(Point(5.0f, 10.0f, 0.0f), Color(0.3, 0.6, 0.1),
      1.5f, 1.0f, 1.0f, 0.7f);

  Sphere *sphere7 = new Sphere(Point(-3.0f, 4.0f, -2.0f), Color(0.1, 0.6, 0.7),
      1.5f, 0.2f, 1.0f, 0.7f);

  Sphere *sphere8 = new Sphere(Point(-4.0f, 7.0f, 3.0f), Color(0.5, 0.1, 0.7),
      1.5f, 0.8f, 1.0f, 0.7f);


  shapes.push_back(floor);
  shapes.push_back(mirror);

  shapes.push_back(sphere0);
  shapes.push_back(sphere1);
  shapes.push_back(sphere2);
  shapes.push_back(sphere3);
  shapes.push_back(sphere4);
  shapes.push_back(sphere5);
  shapes.push_back(sphere6);
  shapes.push_back(sphere7);
  shapes.push_back(sphere8);

  Light *frontLight = new Light(Point(0.0f, 13.0f, 10.0f),
      Color(1.0f, 1.0f, 1.0f), 1.0f);

  lights.push_back(frontLight);

}

void writePPMFile(Image *image, const char *filename, float width, float height)
{
  std::ofstream ofs(filename, std::ios::out | std::ios::binary);
  ofs << "P6\n" << width << " " << height << "\n255\n";
  for (unsigned i = 0; i < width * height; ++i)
  {
    Image pixel = image[i];

    ofs << (unsigned char) (std::min(float(1), pixel.r()) * 255)
      << (unsigned char) (std::min(float(1), pixel.g()) * 255)
      << (unsigned char) (std::min(float(1), pixel.b()) * 255);
  }
  ofs.close();
}

#endif
