/*
 * Ray.h
 *
 *  Created on: Jun 5, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef RAY_H_
#define RAY_H_

#include "Vector3D.h"

// Don't ever start a ray exactly where you previously hit; you must offset it
// a little bit so you don't accidentally 'self-intersect'.
const float kRayTMin = 0.00001f;
// Unless otherwise specified, rays are defined to be able to hit anything as
// far as the computer can see.  You can limit a ray's max if you need to though
// as is done often when calculating shadows, so you only check the range from
// the point on the surface to the point on the light.
const float far = 1.0e30f;

class Ray
{
public:
	Ray ()
			: _origin (), _direction (0.0f, 0.0f, 1.0f), _farPlane (far)
	{

	}

	Ray (const Ray& r)
			: _origin (r.origin()), _direction (r.direction()), _farPlane (r.farPlane())
	{

	}

	Ray (const Point& origin, const Vector3D& direction, float tMax = far)
			: _origin (origin), _direction (direction), _farPlane (tMax)
	{

	}

	Ray& operator = (const Ray& r)
	{
		_origin = r.origin();
		_direction = r.direction();
		_farPlane = r.farPlane();
		return *this;
	}

	inline Point origin () const { return _origin; }
	inline Vector3D direction () const { return _direction; }
	inline float farPlane () const { return _farPlane; }

	inline Point calculate (float t) const
	{
		return _origin + t * _direction ;
	}

private:
	Point _origin;
	Vector3D _direction;
	float _farPlane;
};

#endif /* RAY_H_ */
