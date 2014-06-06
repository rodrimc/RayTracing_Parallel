/*
 * Plane.h
 *
 *  Created on: Jun 5, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef PLANE_H_
#define PLANE_H_

#include "Vector3D.h"
#include "IShape.h"
#include "Color.h"

#include <cmath>

class Plane : public IShape
{
public:
	Plane (const Point& position, const Vector3D& normal, const Color& color,
	     	const float &refl = 0, const float &transp = 0)
			: IShape(position, color, refl, transp), _normal (normal.normalized ())
	{

	}

	virtual ~Plane ()
	{
	}

	virtual bool intersect (Intersection& intersection)
	{
		float nDotD = _normal.dot (intersection.ray.direction ());
		if (nDotD >= 0.0f)
		{
			return false;
		}

		float t = (_position.dot (_normal)
				- intersection.ray.origin ().dot (_normal))
				/ intersection.ray.direction ().dot (_normal);

		if (t >= intersection.t || t < kRayTMin)
		{
			return false;
		}

		intersection.t = t;
		intersection.pShape = this;
		intersection.color = _color;
		intersection.normal = _normal;

		if (std::fmod ((intersection.position () - _position).length () * 0.25f,
											1.0f) > 0.5f)
		{
			intersection.color *= 0.2f;
		}

		return true;
	}

protected:
	Vector3D _normal;
};

#endif /* PLANE_H_ */
