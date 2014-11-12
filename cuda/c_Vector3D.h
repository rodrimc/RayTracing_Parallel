/*
 * Vector3D.h
 *
 *  Created on: Jun 5, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef C_VECTOR3D_H_
#define C_VECTOR3D_H_

#include <cmath>

class Vector3D
{
public:
	__host__ __device__ Vector3D ()
			: _x (0.0f), _y (0.0f), _z (0.0f)
	{
	}
	__host__ __device__ Vector3D (const Vector3D& v)
			: _x (v.x ()), _y (v.y ()), _z (v.z ())
	{
	}
  __host__ __device__ 	Vector3D (float x, float y, float z)
			: _x (x), _y (y), _z (z)
	{
	}

  explicit __host__ __device__ Vector3D (float f)
			: _x (f), _y (f), _z (f)
	{
	}

	inline __host__ __device__ float length2 () const
	{
		return _x * _x + _y * _y + _z * _z;
	}
	inline __host__ __device__ float length () const
	{
		return std::sqrt (length2 ());
	}

	// Returns old length from before normalization (ignore the return value if you don't need it)
	inline __host__ __device__ float normalize ()
	{
		float len = length ();
		*this /= len;
		return len;
	}
	// Return a vector in this same direction, but normalized
	inline __host__ __device__ Vector3D normalized () const
	{
		Vector3D r (*this);
		r.normalize ();
		return r;
	}

	inline float __host__ __device__ x () const
	{
		return _x;
	}
	inline float __host__ __device__ y () const
	{
		return _y;
	}
	inline float __host__ __device__ z () const
	{
		return _z;
	}

	inline __device__ Vector3D& operator = (const Vector3D& v)
	{
		_x = v.x ();
		_y = v.y ();
		_z = v.z ();
		return *this;
	}

	inline __device__ Vector3D& operator += (const Vector3D& v)
	{
		_x += v.x ();
		_y += v.y ();
		_z += v.z ();
		return *this;
	}

	inline __device__ Vector3D& operator -= (const Vector3D& v)
	{
		_x -= v.x ();
		_y -= v.y ();
		_z -= v.z ();
		return *this;
	}

	inline __device__ Vector3D& operator *= (float f)
	{
		_x *= f;
		_y *= f;
		_z *= f;
		return *this;
	}

	inline __host__ __device__ Vector3D& operator /= (float f)
	{
		_x /= f;
		_y /= f;
		_z /= f;
		return *this;
	}

	inline __device__ Vector3D operator - () const
	{
		return Vector3D(-_x, -_y, -_z);
	}

	inline __device__ float dot (const Vector3D& v)
	{
		return _x * v.x () + _y * v.y () + _z * v.z ();
	}

	inline __device__ Vector3D cross (const Vector3D& v)
	{
		return Vector3D (_y * v.z () - _z * v.y (), _z * v.x () - _x * v.z (),
										 _x * v.y () - _y * v.x ());
	}

	inline __device__ void setX (float x)
	{
		_x = x;
	}
	inline __device__ void setY (float y)
	{
		_y = y;
	}
	inline __device__ void setZ (float z)
	{
		_z = z;
	}

private:
	float _x;
	float _y;
	float _z;

};

inline __device__ Vector3D operator * (const Vector3D& v, float f)
{
	return Vector3D (f * v.x (), f * v.y (), f * v.z ());
}

inline __device__ Vector3D operator * (float f, const Vector3D& v)
{
	return Vector3D (f * v.x (), f * v.y (), f * v.z ());
}

inline __device__ Vector3D operator + (const Vector3D& v1, const Vector3D& v2)
{
	return Vector3D (v1.x () + v2.x (), v1.y () + v2.y (), v1.z () + v2.z ());
}

inline __device__ Vector3D operator - (const Vector3D& v1, const Vector3D& v2)
{
	return Vector3D (v1.x () - v2.x (), v1.y () - v2.y (), v1.z () - v2.z ());
}

typedef Vector3D Point;

#endif /* C_VECTOR3D_H_ */

