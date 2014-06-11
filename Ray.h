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

const float kRayTMin = 0.00001f;
const float far = 100.0f;

class Ray
{
public:
	Ray ()
			: _origin (), _direction (0.0f, 0.0f, 1.0f), _farPlane (far)
	{
		_direction.normalize();
	}

	Ray (const Ray& r)
			: _origin (r.origin()), _direction (r.direction()),
			  _farPlane (r.farDistance())
	{
	}

	Ray (const Point& origin, const Vector3D& direction, float tMax = far)
			: _origin (origin), _direction (direction), _farPlane (tMax)
	{
		_direction.normalize();
	}

	Ray& operator = (const Ray& r)
	{
		_origin = r.origin();
		_direction = r.direction();
		_farPlane = r.farDistance();
		return *this;
	}

	inline Point origin () const { return _origin; }
	inline Vector3D direction () const { return _direction; }
	inline float farDistance () const { return _farPlane; }

	inline void setOrigin (const Point &origin) { _origin = origin; }
	inline void setDirection (const Point &direction) { _direction = direction; }

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
