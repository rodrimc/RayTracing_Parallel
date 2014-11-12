
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
#include <unistd.h>

#define MAX_DEPTH 8

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaPeekAtLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
          msg, cudaGetErrorString(__err), \
          __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      return 0; \
    } \
  } while (0)

static const float ambient_coefficient = 0.2f;
static const float bias = 1e-4;

typedef Color Image;

enum intersection_type
{
  NONE, PLANE, SPHERE
};

typedef struct
{
  Ray ray;
  float intensity;
} Task;

  __device__ 
Color ambientColor (const Color& color)
{
  return ambient_coefficient * color;
}

  __device__
Color specularColor (const Vector3D &direction, const Vector3D &normal,
    const Ray& ray, const Light* light,
    const float &specular_coefficient)
{
  Color specular_color;
  Vector3D refl = direction - normal * 2 * Vector3D (direction).dot (normal);
  refl.normalize ();

  float m = refl.dot (ray.direction ());
  m = m < 0 ? 0.0f : m;

  float cosB = m / refl.length () * ray.direction ().length ();
  float spec = pow (cosB, 50);

  specular_color = specular_coefficient * light->color () * spec;

  return specular_color;
}

  __device__
Color diffuseColor (const Vector3D& direction, const Light *light,
    const Vector3D& normal, const Color& color,
    const float &diffuse_coefficient)
{
  Color diffuse_color;
  float dot = Vector3D (normal).dot (direction);
  dot = dot < 0 ? 0.0f : dot;

  diffuse_color = diffuse_coefficient * color * light->color () * dot;

  return diffuse_color;
}

  __device__ 
intersection_type calculateIntersect (const Ray &ray,
    Plane *d_planes, int num_planes,
    Sphere *d_spheres, int num_spheres,
    float &t, Plane &i_plane, Sphere &i_sphere,
    Vector3D &shapeNormal, 
    Color &pixelColor)
{
  t = INFINITY;
  Color color;
  Vector3D normal;
  intersection_type type = NONE;

  for (int i = 0; i < num_planes; i++)
  {
    Plane plane = d_planes[i];
    float near;
    if (plane.intersect (ray, &near, normal, color) && near < t)
    {
      shapeNormal = normal;
      pixelColor = color;
      t = near;
      i_plane = plane;
      type = PLANE;
    }
  }

  for (int i = 0; i < num_spheres; i++)
  {
    Sphere sphere = d_spheres[i];
    float near;
    if (sphere.intersect (ray, &near, normal, color) && near < t)
    {
      shapeNormal = normal;
      pixelColor = color;
      t = near;
      i_sphere = sphere;
      type = SPHERE;
    }
  }

  return type;
}

  __device__ 
Color compute_pixelcolor (Ray first_ray, 
    Plane *d_planes, int num_planes,
    Sphere *d_spheres, int num_spheres,
    Light * d_lights, int num_lights,
    int depth)
{
  Color pixelcolor (0.0f);
  Task tasks [MAX_DEPTH];

  Task t0;
  t0.ray = first_ray;
  t0.intensity = 1.0;
  tasks [0] = t0;

  int tasks_count = 1;

  //I implemented an iteractive version of raytracing, once
  //I've tried the recursive version and it doesn't work.
  //Appeerantly recursion in gpu it is still an issue to be solved.
  do
  {
    depth++;
    
    if (tasks_count == 0)
      break;

    Task current_task = tasks[--tasks_count];
    Ray ray = current_task.ray;

    Color intersection_color;
    Color color(0.3f);
    float near;
    Vector3D normal;

    //Unfortunately it seems that it's not possible to use polymorphism
    //on gpu. That's why I need two variables instead of just one pointer
    //to an IShape object.
    //The 'i_' is a short for intersection.
    Plane i_plane;
    Sphere i_sphere;

    intersection_type type = calculateIntersect (ray, d_planes, num_planes,
        d_spheres, num_spheres, near, i_plane, i_sphere, normal, intersection_color);

    if (type != NONE)
    {
      float intersection_reflection = type == PLANE ? 
        i_plane.reflection () : i_sphere.reflection();

      Point intersection_point = ray.calculate (near);
      Vector3D n;
      Color c;   
      color = Color (0.0f);

      for (int i = 0; i < num_lights; i++)
      {
        Vector3D light_direction = (d_lights[i].position () - intersection_point);

        float light_lenght = light_direction.normalize ();
        const Ray shadow_ray (intersection_point + normal * bias, light_direction,
            light_lenght);
        near = INFINITY;

        intersection_type shadow_itype = calculateIntersect (shadow_ray, d_planes, num_planes,
            d_spheres, num_spheres, near, i_plane, i_sphere, n, c);

        //There is no object between the intersected pixel and this light.
        if (shadow_itype == NONE) 
        {
          float diffuse_coefficient = type == PLANE ? 
            i_plane.diffuse () : i_sphere.diffuse();
          
          float specular_coefficient = type == PLANE ? 
            i_plane.specular () : i_sphere.specular();

          color += ambientColor (intersection_color);
          if (diffuse_coefficient > 0.0f)
            color += diffuseColor (light_direction, &d_lights[i], normal, intersection_color,
                diffuse_coefficient);

          if (specular_coefficient > 0.0f)
            color += specularColor (light_direction, normal, ray, &d_lights[i],
                specular_coefficient);
        }
        else //Intersected pixel is shadowed!!!
        {
          color = intersection_color * 0.1;
          break;
        }
      }

      //Calculate the reflected color
      if (intersection_reflection > 0.0f)
      {
        Task new_task;
        //If this ray is a reflection ray, it must mutiply the shape
        //reflection coefficient by the reflection coefficient that
        //generated this ray
        new_task.intensity = intersection_reflection * current_task.intensity;

        Vector3D refl_dir = ray.direction ()
          - normal * 2 * ray.direction ().dot (normal);
        refl_dir.normalize ();

        new_task.ray = Ray (intersection_point + normal * bias, refl_dir);
        tasks [tasks_count++] = new_task;
      }
      
      color.clamp();
    }
    
    color = color * current_task.intensity;
    pixelcolor += color;

  } while (depth <= MAX_DEPTH);

  pixelcolor.clamp();
  return pixelcolor;
}

  __global__ 
void k_trace (Image *d_image, 
    Plane *d_planes, int num_planes,
    Sphere *d_spheres, int num_spheres,
    Light *d_lights, int num_lights, 
    float aspect_ratio, float tanFov,
    int width, int height)
{
  int outerOffset = (blockIdx.x * gridDim.y + blockIdx.y) 
    * (blockDim.x * blockDim.y);
  int innerOffset = threadIdx.x * blockDim.y + threadIdx.y;
  int final_offset = outerOffset + innerOffset;

  if (final_offset < width * height)
  {
    int y = final_offset / width;
    int x = final_offset % width;
    float yu = (1 - 2 * ((y + 0.5) * 1 / float(height))) * tanFov;
    float xu = (2 * ((x + 0.5) * 1 / float (width)) - 1) * tanFov * aspect_ratio;
    Point origin (0.0f, 5.0f, 20.0f);
    Ray ray (origin, Vector3D (xu, yu, -1));

    d_image[final_offset] = compute_pixelcolor(ray, d_planes, num_planes,
        d_spheres, num_spheres, d_lights, num_lights, 0);
  }
}

bool c_initScene (Sphere **spheres, int *num_spheres, 
    Plane **planes, int *num_planes, 
    Light **lights, int *num_lights)
{
  *num_planes = 2;
  *num_spheres = 9;
  *num_lights = 1;

  Sphere *h_spheres;
  Plane *h_planes;
  Light *h_lights;

  int error = 0;
  //Allocation of memory for arrays on host
  h_spheres = (Sphere*) malloc (sizeof (Sphere) * (*num_spheres));
  if (!h_spheres) error = 1;

  h_planes = (Plane*) malloc (sizeof (Plane) * 2);
  if (!h_planes) error = 1;

  h_lights = (Light*) malloc (sizeof (Light) * (*num_lights));
  if (!h_lights) error = 1;

  if (error)
  {
    if (h_spheres) free (h_spheres);
    if (h_planes) free (h_planes);
    if (h_lights) free (h_lights);

    return false;
  }

  //Allocation of memory for arrays on device 
  cudaMalloc ((void **)&(*planes), sizeof (Plane) * (*num_planes));
  cudaCheckErrors ("Planes array allocation");

  cudaMalloc ((void **)&(*spheres), sizeof (Sphere) * (*num_spheres));
  cudaCheckErrors ("Spheres array allocation");

  cudaMalloc ((void **)&(*lights), sizeof (Light) * (*num_lights));
  cudaCheckErrors ("Lights array allocation");

  //Instantiation of objects
  Plane mirror (Point(0.0f, 0.0, -15.0f), Vector3D (0.0f, 0.0f, 1.0f), 
      Color(0.2f, 0.2f, 0.2f), 1.0f, 0.2f, 0.2f, false);

  Plane floor (Point(0.0f, -2.5f, 0.0f), Vector3D(0.0f, 1.0f, 0.0f),
      Color(1.0f, 1.0f, 1.0f), 0.5f, 0.3f, 0.3f);

  memcpy (&h_planes[0], &mirror, sizeof (Plane));
  memcpy (&h_planes[1], &floor, sizeof (Plane));

  Sphere sphere0 (Point(-5.0f, 1.0f, 0.0f), Color(0.5, 0.1, 0.1),
      0.5f, 0.5f, 1.0f, 0.6f);
  Sphere sphere1 (Point(3.0f, 0.0f, 0.0f), Color(1.0, 0.1, 0.1),
      1.5f, 0.8f, 1.0f, 0.2f);
  Sphere sphere2 (Point(-3.0f, 0.0f, 0.0f), Color(0.1, 1.0, 0.1),
      1.5f, 0.5f, 1.0f, 0.7f);
  Sphere sphere3 (Point(0.0f, 0.0f, -4.0f), Color(0.5, 0.5, 0.5),
      1.5f, 0.5f, 1.0f, 0.7f);
  Sphere sphere4 (Point(10.0f, 5.0f, -4.0f), Color(1.0, 0.3, 0.8),
      1.5f, 0.8f, 1.0f, 0.4f);
  Sphere sphere5 (Point(8.0f, 0.0f, 4.0f), Color(0.5, 0.5, 1.0),
      1.5f, 0.3f, 1.0f, 0.7f);
  Sphere sphere6 (Point(5.0f, 10.0f, 0.0f), Color(0.3, 0.6, 0.1),
      1.5f, 1.0f, 1.0f, 0.7f);
  Sphere sphere7 (Point(-3.0f, 4.0f, -2.0f), Color(0.1, 0.6, 0.7),
      1.5f, 0.2f, 1.0f, 0.7f);
  Sphere sphere8 (Point(-4.0f, 7.0f, 3.0f), Color(0.5, 0.1, 0.7),
      1.5f, 0.8f, 1.0f, 0.7f);

  memcpy (&h_spheres[0], &sphere0, sizeof (Sphere));
  memcpy (&h_spheres[1], &sphere1, sizeof (Sphere));
  memcpy (&h_spheres[2], &sphere2, sizeof (Sphere));
  memcpy (&h_spheres[3], &sphere3, sizeof (Sphere));
  memcpy (&h_spheres[4], &sphere4, sizeof (Sphere));
  memcpy (&h_spheres[5], &sphere5, sizeof (Sphere));
  memcpy (&h_spheres[6], &sphere6, sizeof (Sphere));
  memcpy (&h_spheres[7], &sphere7, sizeof (Sphere));
  memcpy (&h_spheres[8], &sphere8, sizeof (Sphere));

  Light frontLight (Point(0.0f, 13.0f, 10.0f), Color(1.0f, 1.0f, 1.0f), 1.0f);

  memcpy (&h_lights[0], &frontLight, sizeof (Light));

  //Copy of objects from host to device
  //Planes
  cudaMemcpy ((*planes), h_planes, sizeof (Plane) * (*num_planes), cudaMemcpyHostToDevice);
  cudaCheckErrors ("Copying 'h_planes' array to device");

  //Spheres
  cudaMemcpy ((*spheres), h_spheres, sizeof (Sphere) * (*num_spheres), cudaMemcpyHostToDevice);
  cudaCheckErrors ("Copying 'h_spheres' array to device");

  //Lights
  cudaMemcpy ((*lights), h_lights, sizeof (Light) * (*num_lights), cudaMemcpyHostToDevice);
  cudaCheckErrors ("Copying 'h_lights' array to device");


  free (h_planes);
  free (h_spheres);
  free (h_lights);

  if (cudaGetLastError() != cudaSuccess)
    return false;

  return true;
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
