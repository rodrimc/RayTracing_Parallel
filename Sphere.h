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
	Sphere(const Vector3D &position, const Color &color, const float &radius,
			const float &refl = 0)
			: IShape(position, color, refl), _radius(radius)
	{
	}

	virtual ~Sphere()
	{
	}

	virtual bool intersect (const Ray &ray, float *t, Vector3D& normal, Color &pixelColor)
	{
		Ray localRay = ray;
		localRay.setOrigin(localRay.origin() - _position);

		float a = localRay.direction().length2();
		float b = 2.0f * localRay.direction().dot(localRay.origin());
		float c = localRay.origin().length2() - _radius * _radius;

		float t0, t1, discriminant;

		discriminant = b * b - 4 * a * c;
		if (discriminant < 0)
			return false;
		else if (discriminant == 0)
			t0 = t1 = -0.5 * b / a;
		else
		{
			float q =
					(b > 0) ?
							-0.5 * (b + sqrt(discriminant)) :
							-0.5 * (b - sqrt(discriminant));
			t0 = q / a;
			t1 = c / q;
		}
		if (t0 > t1)
			std::swap(t0, t1);

		if (t0 < kRayTMin)
		{
			return false;
		}

		*t = t0;
		normal = (ray.calculate(t0) - _position).normalized();
		pixelColor = _color;

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
