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

#define MAX_DEPTH 7

typedef Color Image;

#define MIX 0.5

float mix (const float &a, const float &b)
{
	return b * MIX + a * (float (1) - MIX);
}

static const int width = 512;
static const int height = 512;

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

bool calculateIntersect (std::set<IShape*> &sceneShapes,
												 Intersection& intersection)
{
	bool isIntersected = false;

	for (auto shape : sceneShapes)
	{
		if (shape->intersect (intersection))
		{
			isIntersected = true;
			break;
		}
	}

	return isIntersected;
}

Color trace (const Ray& ray, std::set<IShape*>& sceneShapes,
						 std::set<Light*>& sceneLights, int depth)
{
	Intersection intersection (ray);

	Color pixelColor (1.0f, 1.0f, 1.0f);
	if (calculateIntersect (sceneShapes, intersection))
	{
		pixelColor += intersection.color;
		IShape *shape = intersection.pShape;
		Point intersectionPoint = intersection.position ();

		if ((shape->transparency () > 0 || shape->reflection () > 0)
				&& depth <= MAX_DEPTH)
		{
			Vector3D normal = intersection.normal;

			float ratio = -ray.direction ().dot (normal);
			float fresnel = mix (pow (1 - ratio, 3), 1);

			Vector3D reflDir = ray.direction () -
					normal * 2 * ray.direction ().dot (normal);
			reflDir.normalize ();

			Ray reflectionRay (intersectionPoint + normal, reflDir);

			Color reflectionColor = trace (reflectionRay, sceneShapes, sceneLights,
																		 depth + 1);

			Color refractionColor;

//			pixelColor = (reflectionColor * fresnel * shape->reflection ()
//					+ refractionColor * (1 - fresnel) * shape->transparency ())
//					* intersection.color;
			pixelColor = reflectionColor * shape->reflection () * shape->color ();
		}

		else
		{
			pixelColor = intersection.color;
			Color lightContribution ;

			for (auto light : sceneLights)
			{
				Vector3D lightDirection =
						(light->position () - intersectionPoint).normalized ();

				Ray shadowRay (intersectionPoint, lightDirection);
				Intersection shadowIntersection (shadowRay);

				if (!calculateIntersect (sceneShapes, shadowIntersection))
				{

					lightContribution += light->color() *
					                        intersection.color *
					                        std::max(0.0f, lightDirection.dot(
					                        		intersection.normal));
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

	Plane plane (Point (0.0f, -2.0f, 0.0f), Vector3D (0.0f, 1.0f, 0.0f),
							 Color (1.0f, 1.0f, 1.0f));

	Sphere sphere1 (Point (3.0f, 0.0f, 0.0f), Color (1.0, 0.5, 0.5), 1.5f, 1.0);
	Sphere sphere2 (Point (-3.0f, 0.0f, 0.0f), Color (0.5, 1.0, 0.5), 1.5f, 1.0);
	Sphere sphere3 (Point (0.0f, 0.0f, 4.0f), Color (0.5, 0.5, 1.0), 1.5f, 1.0);
	Sphere sphere4 (Point (0.0f, 0.0f, -4.0f), Color (0.5, 0.5, 0.5), 1.5f, 1.0);

	sceneShapes.insert (&plane);
	sceneShapes.insert (&sphere1);
	sceneShapes.insert (&sphere2);
	sceneShapes.insert (&sphere3);
	sceneShapes.insert (&sphere4);

	Light areaLight (Point (0.0f, 3.0f, 0.0f), Color (1.0f, 1.0f, 1.0f), 3.0f);
	Light smallAreaLight (Point (0.0f, 1.0f, -2.0f), Color (1.0f, 1.0f, 1.0f),
												3.0f);

	sceneLights.insert (&areaLight);
	sceneLights.insert (&smallAreaLight);

	float fov = 30.0;
	float tanFov = tan (fov * M_PI / 180.0f);

	Image *image = new Image[width * height];

	Point origin (0.0f, 10.0f, 30.0f);
	Point target (0.0f, 0.0f, 1.0f);
	Vector3D targetUpDirection (0.0f, 1.0f, 0.0f);
	Vector3D forward = (target - origin).normalized ();

	Vector3D right = forward.cross (targetUpDirection).normalized ();
	Vector3D up = right.cross (forward).normalized ();

	for (int y = 0; y < height; y++)
	{
		float yu = 1.0f - (float (y) / float (height - 1));
		for (int x = 0; x < width; x++)
		{
			float xu = float (x) / float (width - 1);

			Ray ray (
					origin,
					forward + right * ((xu - 0.5f) * tanFov)
							+ up * ((yu - 0.5f) * tanFov));

			image[y * width + x] = trace (ray, sceneShapes, sceneLights, 0);
		}
	}

	writePPMFile (image, filename.c_str (), width, height);

	system (std::string ("eog " + filename).c_str ());

	return 1;
}
