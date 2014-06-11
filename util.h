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

static const int width = 800;
static const int height = 600;

static const float airRefracIndex = 1.0f;
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
	Plane *floor = new Plane(Point(0.0f, -2.0f, 0.0f), Vector3D(0.0f, 1.0f, 0.0f),
			Color(1.0f, 1.0f, 1.0f), 0.0f, 0.0f, 0.0f, 1.0f, 0.7f);

	Plane *backMirror = new Plane(Point(0.0f, 0.0f, -20.0f),
			Vector3D(0.0f, 0.0f, 1.0f), Color(0.5f, 0.5f, 0.5f), 1.0, 0.0f, 0.0, 0.0,
			0.0f, false);

	Sphere *sphere1 = new Sphere(Point(3.0f, 0.0f, 0.0f), Color(1.0, 0.1, 0.1),
			1.5f, 1.0, 0.0f, 0.0f, 1.0f, 0.7f);
	Sphere *sphere2 = new Sphere(Point(-3.0f, 0.0f, 0.0f), Color(0.1, 1.0, 0.1),
			1.5f, 0.0, 0.0f, 0.0f, 1.0f, 0.7f);
	Sphere *sphere3 = new Sphere(Point(0.0f, 0.0f, 4.0f), Color(0.1, 0.1, 1.0),
			1.5f, 0.0, 0.0f, 0.0f, 1.0f, 0.7f);
	Sphere *sphere4 = new Sphere(Point(0.0f, 0.0f, -4.0f), Color(0.5, 0.5, 0.5),
			1.5f, 0.0, 0.0f, 0.0f, 1.0f, 0.7f);

	Sphere *sphere5 = new Sphere(Point(0.0f, 10.0f, 0.0f), Color(1.0, 1.0, 1.0),
			3.0f, 1.0, 0.0f, 0.0f, 0.0f, 0.0f);

	sceneShapes.insert(floor);
	sceneShapes.insert(backMirror);

	sceneShapes.insert(sphere1);
	sceneShapes.insert(sphere2);
	sceneShapes.insert(sphere3);
	sceneShapes.insert(sphere4);
	sceneShapes.insert(sphere5);

	Light *leftLight = new Light(Point(0.0f, 3.0f, -3.0f),
			Color(1.0f, 1.0f, 1.0f), 1.0f);
	Light *rightLight = new Light(Point(0.0f, 3.0f, 3.0f),
			Color(1.0f, 1.0f, 1.0f), 1.0f);

	sceneLights.insert(leftLight);
	sceneLights.insert(rightLight);
}
#endif /* UTIL_H_ */
