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
				 const float &refl = 0, bool flag = true)
			: IShape (position, color, refl), _normal (normal.normalized ()),
				_flag (flag)
	{

	}

	virtual ~Plane ()
	{
	}

	virtual bool intersect (const Ray &ray, float *t, Vector3D& normal,
													Color &pixelColor)
	{
		float nDotD = _normal.dot (ray.direction ());
		if (nDotD >= 0.0f)
		{
			return false;
		}

		float t0 = (_position.dot (_normal) - ray.origin ().dot (_normal))
				/ ray.direction ().dot (_normal);

		if (t0 < kRayTMin)
		{
			return false;
		}

		*t = t0;
		normal = _normal;
		pixelColor = _color;

		if (_flag
				&& std::fmod ((ray.calculate (t0) - _position).length () * 0.25f, 1.0f)
						> 0.5f)
		{
			pixelColor *= 0.2f;
		}

		return true;
	}

protected:
	Vector3D _normal;
	bool _flag;
};

#endif /* PLANE_H_ */
