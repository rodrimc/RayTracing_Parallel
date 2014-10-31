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

#include <iostream>
#include <fstream>

typedef Color Image;

static const float ambient_coefficient = 0.2f;
static const float bias = 1e-4;

void writePPMFile(Image *image, const char *filename, float width, float height);

void initScene(std::set<IShape *>&sceneShapes, std::set<Light *>&sceneLights);

IShape* calculateIntersect (const Ray &ray, std::set<IShape*> &sceneShapes,
														float *t, Vector3D &shapeNormal, Color &pixelColor);


Color ambientColor (const Color& color);


Color specularColor (const Vector3D &direction, const Vector3D &normal,
										 const Ray& ray, const Light* light,
										 const float &specularCoefficient);


Color diffuseColor (const Vector3D& direction, const Light *light,
										const Vector3D& normal, const Color& color,
										const float &diffuseCoefficient);


Color trace (const Ray& ray, std::set<IShape*>& sceneShapes,
						 std::set<Light*>& sceneLights, int depth);

#endif /* UTIL_H_ */
