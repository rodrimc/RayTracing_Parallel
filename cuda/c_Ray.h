/*
 * c_Ray.h
 *
 *  Created on: Oct 31, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef C_RAY_H_
#define C_RAY_H_

#include "c_Vector3D.h"

const float kRayTMin = 0.00001f;
const float far = 60.0f;

class Ray
{
public:
	__device__ Ray ()
			: _origin (), _direction (0.0f, 0.0f, 1.0f), _farPlane (far)
	{
		_direction.normalize();
	}

	__device__ Ray (const Ray& r)
			: _origin (r.origin()), _direction (r.direction()),
			  _farPlane (r.farDistance())
	{
	}

	__device__ Ray (const Point& origin, const Vector3D& direction, float tMax = far)
			: _origin (origin), _direction (direction), _farPlane (tMax)
	{
		_direction.normalize();
	}

	__device__ Ray& operator = (const Ray& r)
	{
		_origin = r.origin();
		_direction = r.direction();
		_farPlane = r.farDistance();
		return *this;
	}

	inline __device__ Point origin () const { return _origin; }
	inline __device__ Vector3D direction () const { return _direction; }
	inline __device__ float farDistance () const { return _farPlane; }

	inline __device__ void setOrigin (const Point &origin) { _origin = origin; }
	inline __device__ void setDirection (const Point &direction) { _direction = direction; }

	inline __device__ Point calculate (float t) const
	{
		return _origin + t * _direction ;
	}

private:
	Point _origin;
	Vector3D _direction;
	float _farPlane;
};

#endif /* C_RAY_H_ */

