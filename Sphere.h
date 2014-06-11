/*
 * Sphere.h
 *
 *  Created on: Jun 6, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef SPHERE_H_
#define SPHERE_H_

#include "IShape.h"
#include "Vector3D.h"

class Sphere: public IShape
{
public:
	Sphere (const Vector3D &position, const Color &color, const float &radius,
			const float &refl = 0.0f, const float &transp = 0.0f,  float refracIndex =
					0.0f, float spec = 0.0f, float diff = 0.0f)
	: IShape (position, color, refl, transp, refracIndex, spec, diff), _radius (radius)
	{
	}

	virtual ~Sphere()
	{
	}

	virtual bool intersect(const Ray &ray, float *t, Vector3D& normal,
			Color &pixelColor)
	{
		Point line = _position - ray.origin();
		float tca = line.dot(ray.direction());

		if (tca < 0)
			return false;

		float d2 = line.length2() - tca * tca;

		if (d2 > _radius * _radius)
			return false;

		float thc = sqrt(_radius * _radius - d2);

		float t0 = tca - thc;
		float t1 = tca + thc;

		if (t0 > t1)
			std::swap(t0, t1);

		if (t0 > ray.farDistance())
			return false;

		*t = t0;
		normal = (ray.calculate(t0) - _position).normalized();
		pixelColor = _color;

		return true;

		return true;
	}

	float radius() const
	{
		return _radius;
	}

protected:
	float _radius;

};

#endif /* SPHERE_H_ */
