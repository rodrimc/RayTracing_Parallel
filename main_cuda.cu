/*
 * main_cuda.cu
 *
 *  Created on: Oct 30, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#include "c_util.h"

#include <math.h>
#include <stdio.h>
#include <string>
#include <sys/time.h>

#define MAX_DEPTH 8

int main (int argc, char** argv)
{
  int num_bytes;
  timeval t_start, t_end;
  double elapsed_time;
	Image *h_image, *d_image;
  thrust::host_vector<IShape *> h_shapes;
  thrust::host_vector<Light *> h_lights;
  thrust::device_vector<IShape *> d_shapes;
  thrust::device_vector<Light *> d_lights;

  dim3 threadsPerBlock (16, 16);
  dim3 numBlocks;

	if (argc < 3)
	{
		printf ("Usage: %s <widht> <height> [<fov>]\n", argv[0]);
		return 0;
	}

	int width = atoi (argv[1]);;
	int height = atoi (argv[2]);;

  num_bytes = (width * height) * sizeof(Color);

	float fov = 60.0;
	if (argc >= 4)
	{
		fov = atof (argv[3]);
	}

  h_image = new Image[width * height];
  
	c_initScene (h_shapes, h_lights);
  
  //Allocate array on device
  cudaMalloc (&d_image, num_bytes);
  
  //Copy data host -> device
  d_shapes = h_shapes;
  d_lights = h_lights;
  cudaMemcpy (h_image, d_image, num_bytes, cudaMemcpyHostToDevice);

  float tanFov = tan (fov * 0.5 * M_PI / 180.0f);

	float aspect_ratio = float (width) / float (height);
	Point origin (0.0f, 5.0f, 20.0f);

	std::string filename = "out.ppm";

	printf ("Rendering scene:\n");
	printf ("Width: %d \nHeight: %d\nFov: %.2f\n", width, height, fov);

  numBlocks = dim3 (width/threadsPerBlock.x + 1, height/threadsPerBlock.y + 1);

  printf ("Blocks: %d x %d\n", numBlocks.x, numBlocks.y);

  gettimeofday (&t_start, NULL);
  
  k_trace <<<numBlocks, threadsPerBlock>>> (d_image, width, height, aspect_ratio,
      tanFov, 0);

  gettimeofday (&t_end, NULL);
  
  elapsed_time = (t_end.tv_sec - t_start.tv_sec) * 1000.0;
  elapsed_time += (t_end.tv_usec - t_start.tv_usec) / 1000.0;

  printf ("\r100.00%%");
	printf ("\nFinished!\n");
  printf ("Rendering time: %.3f s\n", elapsed_time/1000.0);

  /*
	for (int y = 0; y < height; y++)
	{
		printf ("\r%.2f%%", float(y)/height * 100);

		for (int x = 0; x < width; x++)
		{
      Color color;
			float yu = (1 - 2 * ((y + 0.5) * 1 / height)) * tanFov;
			float xu = (2 * ((x + 0.5) * 1 / float (width)) - 1) * tanFov
					* aspect_ratio;
			Ray ray (origin, Vector3D (xu, yu, -1));

			color = trace (ray, shapes, sceneLights, 0);
      
      h_image[y * width + x] = color;
		}

	}
  printf ("\r100.00%%");
	printf ("\nFinished!\n");

  writePPMFile (h_image, filename.c_str (), width, height);

	std::set<IShape*>::iterator it = shapes.begin ();
	while (it != shapes.end ())
	{
		free (*it);
		shapes.erase (it);
		it++;
	}

	std::set<Light*>::iterator it2 = sceneLights.begin ();
	while (it2 != sceneLights.end ())
	{
		free (*it2);
		sceneLights.erase (it2);
		it2++;
	}
  */

  cudaMemcpy (h_image, d_image, num_bytes, cudaMemcpyDeviceToHost);
  
/*
  printf ("\n");
  for (int i = 0; i < width; i++)
  {
    for (int j = 0; j < height; j++)
    {
      int index = i * width + j;
      Color c = h_image[index];
      printf ("%.2f,%.2f,%.2f ", c.r(), c.g(), c.b());
    } 
    printf ("\n");
  }
  */
  
  writePPMFile (h_image, filename.c_str(), width, height);


  delete h_image;
  cudaFree (d_image);
  
	return 0;
}
