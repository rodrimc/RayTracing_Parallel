/*
 * main.cpp
 *
 *  Created on: Jun 5, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#include "IShape.h"
#include "Vector3D.h"
#include "Color.h"
#include "Light.h"
#include "util.h"

#include <set>
#include <math.h>
#include <string>
#include <iostream>

#define MAX_DEPTH 8

extern const float bias;

int main (int argc, char** argv)
{
	if (argc < 3)
	{
		printf ("Usage: ./RayTracerT3 <widht> <height> [<fov>]\n");
		return 0;
	}

	int width = atoi (argv[1]);;
	int height = atoi (argv[2]);;


	float fov = 60.0;
	if (argc >= 4)
	{
		fov = atof (argv[3]);
	}

	std::set<IShape *> sceneShapes;
	std::set<Light *> sceneLights;

	initScene (sceneShapes, sceneLights);

	float tanFov = tan (fov * 0.5 * M_PI / 180.0f);

	float aspectratio = float (width) / float (height);
	Point origin (0.0f, 5.0f, 20.0f);

#ifdef  SDL_SUPPORT
	SDL_Window *window;
	SDL_Renderer *render;
	SDL_Surface *surface;
	SDL_Texture *texture;
	bool sdlOk = sdlBootstrap (&window, &render, &surface, &texture, width,
			height);
	Uint32 * pixels = new Uint32[width * height];

#endif
	std::string filename = "out.ppm";
	Image *image = new Image[width * height];

	printf ("Rendering scene:\n");
	printf ("Width: %d \nHeight: %d\nFov: %.2f\n", width, height, fov);

	for (int y = 0; y < height; y++)
	{
		printf ("\r%.2f%%", float(y)/height * 100);

#pragma omp parallel for
		for (int x = 0; x < width; x++)
		{
      Color color;
			float yu = (1 - 2 * ((y + 0.5) * 1 / height)) * tanFov;
			float xu = (2 * ((x + 0.5) * 1 / float (width)) - 1) * tanFov
					* aspectratio;
			Ray ray (origin, Vector3D (xu, yu, -1));

			color = trace (ray, sceneShapes, sceneLights, 0);
      
#ifdef  SDL_SUPPORT
			if (sdlOk)
			{
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
				pixels[y * width + x] = SDL_MapRGBA (surface->format,
						Uint8 (color.r () * 255),
						Uint8 (color.g () * 255),
						Uint8 (color.b () * 255), Uint8 (255));
#else
				pixels[y * width + x] = SDL_MapRGBA (surface->format,
						Uint8 (color.b () * 255),
						Uint8 (color.g () * 255),
						Uint8 (color.r () * 255),
						Uint8 (255));
#endif
			}
#endif
      image[y * width + x] = color;
		}

#ifdef  SDL_SUPPORT
		if (sdlOk)
		{
#pragma omp critical
      {
        SDL_UpdateTexture (texture, NULL, pixels, width * sizeof(Uint32));
        SDL_RenderClear (render);
        SDL_RenderCopy (render, texture, NULL, NULL);
        SDL_RenderPresent (render);
      }
		}
#endif
	}
  printf ("\r100.00%%");
	printf ("\nFinished!\n");

#ifdef  SDL_SUPPORT
	if (sdlOk)
	{
		SDL_RenderPresent (render);

		bool quit = false;

		SDL_Event event;
		while (!quit)
		{
			SDL_WaitEvent (&event);
			switch (event.type)
			{
				case SDL_QUIT:
				quit = true;
				break;
			}
		}
	}
#endif
  writePPMFile (image, filename.c_str (), width, height);

	std::set<IShape*>::iterator it = sceneShapes.begin ();
	while (it != sceneShapes.end ())
	{
		free (*it);
		sceneShapes.erase (it);
		it++;
	}

	std::set<Light*>::iterator it2 = sceneLights.begin ();
	while (it2 != sceneLights.end ())
	{
		free (*it2);
		sceneLights.erase (it2);
		it2++;
	}

#ifdef  SDL_SUPPORT
	delete[] pixels;

	if (sdlOk)
	{
		SDL_FreeSurface (surface);
		SDL_DestroyTexture (texture);
		SDL_DestroyRenderer (render);
		SDL_DestroyWindow (window);
	}
#endif
	delete image;

	return 0;
}
