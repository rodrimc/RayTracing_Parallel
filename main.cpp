/*
 * main.cpp
 *
 *  Created on: Jun 5, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#include "IShape.h"
#include "Plane.h"
#include "Vector3D.h"
#include "Color.h"
#include "Light.h"
#include "Sphere.h"

#include <set>
#include <math.h>
#include <string>
#include <fstream>
#include <iostream>
#include <SDL2/SDL.h>

#define MAX_DEPTH 7

typedef Color Image;

void logSDLError (std::ostream &os, const std::string &msg)
{
	os << msg << " error: " << SDL_GetError () << std::endl;
}

static const int width = 1280;
static const int height = 720;

void writePPMFile (Image *image, const char *filename, float width,
									 float height)
{
	std::ofstream ofs (filename, std::ios::out | std::ios::binary);
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for (unsigned i = 0; i < width * height; ++i)
	{
		Image pixel = image[i];

		ofs << (unsigned char) (std::min (float (1), pixel.r ()) * 255)
				<< (unsigned char) (std::min (float (1), pixel.g ()) * 255)
				<< (unsigned char) (std::min (float (1), pixel.b ()) * 255);
	}
	ofs.close ();
}

IShape* calculateIntersect (const Ray &ray, std::set<IShape*> &sceneShapes,
														float *t, Vector3D &shapeNormal, Color &pixelColor)
{
	*t = INFINITY;
	Color color;
	Vector3D normal;
	IShape *shapeIntersection = 0;
	for (auto shape : sceneShapes)
	{
		float near;
		if (shape->intersect (ray, &near, normal, color) && near < *t)
		{
			shapeNormal = normal;
			pixelColor = color;
			*t = near;
			shapeIntersection = shape;
		}
	}

	return shapeIntersection;
}

Color trace (const Ray& ray, std::set<IShape*>& sceneShapes,
						 std::set<Light*>& sceneLights, int depth)
{
	Color pixelColor (0.8);
	float near;
	Color color;
	Vector3D normal;
	IShape *shape = calculateIntersect (ray, sceneShapes, &near, normal, color);
	if (shape)
	{
		pixelColor += color;
		Point intersectionPoint = ray.calculate (near);

		if (ray.direction ().dot (normal) > 0) normal = -normal;

		if (shape->reflection () > 0 && depth <= MAX_DEPTH)
		{
			Vector3D reflDir = ray.direction ()
					- normal * 2 * ray.direction ().dot (normal);
			reflDir.normalize ();

			Ray reflectionRay (intersectionPoint + normal, reflDir);
			Color reflectionColor = trace (reflectionRay, sceneShapes, sceneLights,
																		 depth + 1);

			pixelColor = reflectionColor * shape->reflection () * color;
		}
		else
		{
			pixelColor = color;
			Color lightContribution;

			for (auto light : sceneLights)
			{
				Vector3D lightDirection =
						(light->position () - intersectionPoint).normalized ();

				const Ray shadowRay (intersectionPoint, lightDirection);
				float near = INFINITY;
				IShape *s = calculateIntersect (shadowRay, sceneShapes, &near, normal,
																				color);
				if (!s)
				{
					lightContribution += light->color () * pixelColor;
				}
			}
			pixelColor += lightContribution;
		}
		pixelColor.clamp ();
	}

	return pixelColor;
}

int main ()
{
	std::string filename = "out.ppm";

	std::set<IShape *> sceneShapes;
	std::set<Light *> sceneLights;

	Plane floor (Point (0.0f, -2.0f, 0.0f), Vector3D (0.0f, 1.0f, 0.0f),
							 Color (1.0f, 1.0f, 1.0f));

	Plane mirror (Point (0.0f, 0.0f, -20.0f), Vector3D (0.0f, 0.0f, 1.0f),
								Color (0.5f, 0.5f, 0.5f), 0.4, false);

	Sphere sphere1 (Point (3.0f, 0.0f, 0.0f), Color (1.0, 0.5, 0.5), 1.5f, 0.4);
	Sphere sphere2 (Point (-3.0f, 0.0f, 0.0f), Color (0.5, 1.0, 0.5), 1.5f, 1.0);
	Sphere sphere3 (Point (0.0f, 0.0f, 4.0f), Color (0.5, 0.5, 1.0), 1.5f, 1.0);
	Sphere sphere4 (Point (0.0f, 0.0f, -4.0f), Color (0.5, 0.5, 0.5), 1.5f, 1.0);

	sceneShapes.insert (&floor);
	sceneShapes.insert (&mirror);
	sceneShapes.insert (&sphere1);
	sceneShapes.insert (&sphere2);
	sceneShapes.insert (&sphere3);
	sceneShapes.insert (&sphere4);

	Light leftLight (Point (-2.0f, 3.0f, 4.0f), Color (1.0f, 1.0f, 1.0f), 1.0f);
	Light rightLight (Point (2.0f, 3.0f, 4.0f), Color (1.0f, 1.0f, 1.0f), 1.0f);

	sceneLights.insert (&leftLight);
	sceneLights.insert (&rightLight);

	float fov = 30.0;
	float tanFov = tan (fov * M_PI / 180.0f);

	float aspectratio = float (width) / float (height);
	Image *image = new Image[width * height];

	Point origin (0.0f, 5.0f, 20.0f);

	bool sdlOk = SDL_Init (SDL_INIT_EVERYTHING) == 0;
	SDL_Window *window;

	if (!sdlOk)
		logSDLError (std::cout, "SDL_Init failed");
	else window = SDL_CreateWindow ("Ray Tracer", SDL_WINDOWPOS_UNDEFINED,
	SDL_WINDOWPOS_UNDEFINED,
																	width, height, 0);
	if (!window)
	{
		logSDLError (std::cout, "SDL_CreateWindow failed");
		sdlOk = false;
	}

	SDL_Renderer *render = SDL_CreateRenderer (window, -1, 0);

	if (!render)
	{
		logSDLError (std::cout, "SDL_CreateRenderer failed");
		sdlOk = false;
	}
	else
	{
		SDL_RenderClear (render);
	}

	Uint32 rmask, gmask, bmask, amask;

#if SDL_BYTEORDER == SDL_BIG_ENDIAN
	rmask = 0xff000000;
	gmask = 0x00ff0000;
	bmask = 0x0000ff00;
	amask = 0x000000ff;
#else
	rmask = 0x000000ff;
	gmask = 0x0000ff00;
	bmask = 0x00ff0000;
	amask = 0xff000000;
#endif

	SDL_Surface *surface = SDL_CreateRGBSurface (0, width, height, 32, rmask,
																							 gmask, bmask, amask);

	if (!surface)
	{
		logSDLError (std::cout, "SDL_CreateRGBSurface failed");
		sdlOk = false;
	}
	SDL_Texture * texture = SDL_CreateTextureFromSurface (render, surface);
	if (!texture)
	{
		logSDLError (std::cout, "SDL_CreateTextureFromSurface failed");
		sdlOk = false;
	}

	Uint32 * pixels = new Uint32[width * height];

	for (int y = 0; y < height; y++)
	{
		Color color;
		float yu = (1 - 2 * ((y + 0.5) * 1 / height)) * tanFov;
		for (int x = 0; x < width; x++)
		{
			float xu = (2 * ((x + 0.5) * 1 / float (width)) - 1) * tanFov
					* aspectratio;
			Ray ray (origin, Vector3D (xu, yu, -1));
			color = trace (ray, sceneShapes, sceneLights, 0);
			image[y * width + x] = color;

			pixels[y * width + x] = SDL_MapRGBA (surface->format,
																					 Uint8 (color.r () * 255),
																					 Uint8 (color.g () * 255),
																					 Uint8 (color.b () * 255),
																					 Uint8 (255));
		}
		if (sdlOk)
		{
			SDL_UpdateTexture (texture, NULL, pixels, width * sizeof(Uint32));
			SDL_RenderClear (render);
			SDL_RenderCopy (render, texture, NULL, NULL);
			SDL_RenderPresent (render);
		}
	}

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

		delete[] pixels;
	}
	else
	{
		writePPMFile (image, filename.c_str (), width, height);
		system (std::string ("eog " + filename).c_str ());
	}

	SDL_FreeSurface (surface);
	SDL_DestroyTexture (texture);
	SDL_DestroyRenderer (render);
	SDL_DestroyWindow (window);

	return 1;
}
