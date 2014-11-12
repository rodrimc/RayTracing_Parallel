/*
 * c_Plane.h
 *
 *  Created on: Oct 31, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef C_PLANE_H_
#define C_PLANE_H_

#include "c_Vector3D.h"
#include "c_IShape.h"
#include "c_Color.h"

#include <cmath>

class Plane : public IShape
{
public:
	Plane (const Point& position, const Vector3D& normal, const Color& color,
				 const float &refl = 0.0f, float spec = 0.0f, float diff = 0.0f,
				 bool flag = true)
			: IShape (position, color, refl, spec, diff),
				_normal (normal.normalized ()), _flag (flag)
	{
	}

  __device__ Plane ()
    : IShape (Point (0.0, 0.0, 0.0), Color (0.0, 0.0, 0.0), 0, 0, 0)
  {}

	__host__ __device__ virtual ~Plane ()
	{
	}

	virtual __device__ bool intersect (const Ray &ray, float *t, Vector3D& normal,
													Color &pixelColor)
	{
		float nDotD = _normal.dot (ray.direction ());
		if (nDotD >= 0.0f)
		{
			return false;
		}

		float t0 = (_position.dot (_normal) - ray.origin ().dot (_normal))
				/ ray.direction ().dot (_normal);

		if (t0 > ray.farDistance ()) return false;

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

#endif /* C_PLANE_H_ */

