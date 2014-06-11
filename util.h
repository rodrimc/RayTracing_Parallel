/*
 * util.h
 *
 *  Created on: Jun 10, 2014
 *      Author: Rodrigo Costa
 *      email:  rodrigocosta@telemidia.puc-rio.br
 */

#ifndef UTIL_H_
#define UTIL_H_

#include "IShape.h"
#include "Plane.h"
#include "Sphere.h"
#include "Light.h"
#include "Color.h"

#include <set>

#ifdef  SDL_SUPPORT
#include <SDL2/SDL.h>
#endif

#include <iostream>
#include <fstream>

static const float ambientCoefficient = 0.2f;
static const float bias = 1e-4;

typedef Color Image;

void logSDLError(std::ostream &os, const std::string &msg)
{
	os << msg << " error: " << SDL_GetError() << std::endl;
}

#ifdef SDL_SUPPORT
bool sdlBootstrap(SDL_Window **window, SDL_Renderer **render,
		SDL_Surface **surface, SDL_Texture **texture, const int width,
		const int height)
{
	bool sdlOk = SDL_Init(SDL_INIT_EVERYTHING) == 0;

	if (!sdlOk)
		logSDLError(std::cout, "SDL_Init failed");
	else
		*window = SDL_CreateWindow("Ray Tracer", SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED, width, height, 0);
	if (window == nullptr)
	{
		logSDLError(std::cout, "SDL_CreateWindow failed");
		sdlOk = false;
	}

	*render = SDL_CreateRenderer(*window, -1, 0);

	if (render == nullptr)
	{
		logSDLError(std::cout, "SDL_CreateRenderer failed");
		sdlOk = false;
	}
	else
	{
		SDL_RenderClear(*render);
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

	*surface = SDL_CreateRGBSurface(0, width, height, 32, rmask, gmask, bmask,
			amask);
	if (surface == nullptr)
	{
		logSDLError(std::cout, "SDL_CreateRGBSurface failed");
		sdlOk = false;
	}

	*texture = SDL_CreateTextureFromSurface(*render, *surface);
	if (texture == nullptr)
	{
		logSDLError(std::cout, "SDL_CreateTextureFromSurface failed");
		sdlOk = false;
	}

	return sdlOk;
}
#endif

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

void initScene(std::set<IShape *>&sceneShapes, std::set<Light *>&sceneLights)
{
	//new Plane (position, normal, color, reflection, specular coefficient,
	//					diffuse coefficient)

	Plane *floor = new Plane(Point(0.0f, -2.5f, 0.0f), Vector3D(0.0f, 1.0f, 0.0f),
			Color(1.0f, 1.0f, 1.0f), 0.5f, 0.3f, 0.3f);

	//new Sphere (position, color, radius, reflection, specular coefficient,
	//					diffuse coefficient)
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


	sceneShapes.insert(floor);

	sceneShapes.insert(sphere1);
	sceneShapes.insert(sphere2);
	sceneShapes.insert(sphere3);
	sceneShapes.insert(sphere4);
	sceneShapes.insert(sphere5);
	sceneShapes.insert(sphere6);
	sceneShapes.insert(sphere7);
	sceneShapes.insert(sphere8);

	Light *frontLight = new Light(Point(0.0f, 13.0f, 10.0f),
				Color(1.0f, 1.0f, 1.0f), 1.0f);

	sceneLights.insert(frontLight);
}
#endif /* UTIL_H_ */
