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

class Sphere : public IShape
{
public:
	Sphere (const Vector3D &position, const Color &color, const float &radius,
					const float &refl = 0, const float &transp = 0)
			: IShape (position, color, refl, transp), _radius (radius)
	{
	}

	virtual ~Sphere ()
	{
	}

	virtual bool intersect (Intersection& intersection)
	{
		Ray localRay = intersection.ray;
		localRay.setOrigin (localRay.origin () - _position);

//		Vector3D l = _position - localRay.origin ();
//		float tca = l.dot (localRay.direction());
//		if (tca < 0) return false;
//
//		float radius2 = _radius * _radius;
//
//		float d2 = l.dot (l) - tca * tca;
//		if (d2 > radius2) return false;
//
//		float thc = sqrt (radius2 - d2);
//		float t0 = tca - thc;
//		float t1 = tca + thc;

		float a = localRay.direction ().length2 ();
		float b = 2.0f * localRay.direction ().dot (localRay.origin ());
		float c = localRay.origin ().length2 () - _radius * _radius;

		float t0, t1, discriminant;
		discriminant = b * b - 4.0f * a * c;
		if (discriminant < 0.0f)
		{
			return false;
		}
		discriminant = std::sqrt (discriminant);

		t0 = (-b + std::sqrt (discriminant)) * 0.5 * 1 / a;
		t1 = (-b - std::sqrt (discriminant)) * 0.5 * 1 / a;

		float near = t0 < t1 ? t0 : t1;

		if (near >= intersection.t || near < kRayTMin)
		{
			return false;
		}

		intersection.t = near;
		intersection.pShape = this;
		intersection.normal = (intersection.position () - _position).normalized ();
		intersection.color = _color;
		return true;
	}

	float radius () const
	{
		return _radius;
	}

protected:
	float _radius;

};

#endif /* SPHERE_H_ */
