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

#include <set>
#include <math.h>
#include <string>
#include <fstream>

typedef Color Image;

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

int main ()
{
	std::string filename = "out.ppm";

	std::set<IShape *> sceneShapes;
	std::set<Light *> sceneLights;

	Plane plane (Point (0.0f, -2.0f, 0.0f), Vector3D (0.0f, 1.0f, 0.0f),
	             Color(1.0f, 1.0f, 1.0f));

	sceneShapes.insert (&plane);

	RectangleLight areaLight (Point (-2.5f, 5.0f, -2.5f),
														Vector3D (5.0f, 0.0f, 0.0f),
														Vector3D (0.0f, 0.0f, 5.0f),
														Color (0.0f, 0.0f, 1.0f), 3.0f);

	RectangleLight smallAreaLight (Point (0.0f, 1.0f, -2.0f),
																 Vector3D (4.0f, 0.0f, 0.0f),
																 Vector3D (0.0f, 0.0f, 4.0f),
																 Color (1.0f, 0.0f, 0.0f), 3.0f);

	sceneLights.insert (&areaLight);
	sceneLights.insert (&smallAreaLight);


	float fov = 30.0;
	float tanFov = tan (fov * M_PI / 180.0f);

	Image *image = new Image[width * height];

	Point origin (0.0f, 15.0f, 25.0f);
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

			Intersection intersection (ray);

			Color pixelColor (0.0f, 0.0f, 0.0f);
			if (calculateIntersect (sceneShapes, intersection))
			{
				pixelColor += intersection.color;

				Point intersectionPoint = intersection.position ();
				for (auto light : sceneLights)
				{
					Point lightPoint = light->position ();
					Vector3D lightDirection = lightPoint - intersectionPoint;
					Vector3D lightNormal = lightDirection.normalized ();

					float lightDistance = lightDirection.normalize ();

					Ray shadowRay (intersectionPoint, lightDirection, lightDistance);
					Intersection shadowIntersection (shadowRay);

					if (!calculateIntersect(sceneShapes, shadowIntersection))
					{
						// The light point is visible, so let's add that
						// lighting contribution
						float lightAttenuation = std::max (
								0.0f, intersection.normal.dot (lightDirection));
						pixelColor += intersection.color * light->color()
								* lightAttenuation;
					}
				}

			}
			pixelColor.clamp ();

			image[y * width + x] = pixelColor;
		}
	}

	writePPMFile (image, filename.c_str (), width, height);

	system (std::string ("eog " + filename).c_str ());

	return 1;
}
