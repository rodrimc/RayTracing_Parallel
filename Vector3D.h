/*
 * Vector3D.h
 *
 *  Created on: Jun 5, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef VECTOR3D_H_
#define VECTOR3D_H_

#include <cmath>

class Vector3D
{
public:
	Vector3D ()
			: _x (0.0f), _y (0.0f), _z (0.0f)
	{
	}
	Vector3D (const Vector3D& v)
			: _x (v.x ()), _y (v.y ()), _z (v.z ())
	{
	}
	Vector3D (float x, float y, float z)
			: _x (x), _y (y), _z (z)
	{
	}
	explicit Vector3D (float f)
			: _x (f), _y (f), _z (f)
	{
	}

	inline float length2 () const
	{
		return _x * _x + _y * _y + _z * _z;
	}
	inline float length () const
	{
		return std::sqrt (length2 ());
	}

	// Returns old length from before normalization (ignore the return value if you don't need it)
	inline float normalize ()
	{
		float len = length ();
		*this /= len;
		return len;
	}
	// Return a vector in this same direction, but normalized
	inline Vector3D normalized () const
	{
		Vector3D r (*this);
		r.normalize ();
		return r;
	}

	inline float x () const
	{
		return _x;
	}
	inline float y () const
	{
		return _y;
	}
	inline float z () const
	{
		return _z;
	}

	inline Vector3D& operator = (const Vector3D& v)
	{
		_x = v.x ();
		_y = v.y ();
		_z = v.z ();
		return *this;
	}

	inline Vector3D& operator += (const Vector3D& v)
	{
		_x += v.x ();
		_y += v.y ();
		_z += v.z ();
		return *this;
	}

	inline Vector3D& operator -= (const Vector3D& v)
	{
		_x -= v.x ();
		_y -= v.y ();
		_z -= v.z ();
		return *this;
	}

	inline Vector3D& operator *= (float f)
	{
		_x *= f;
		_y *= f;
		_z *= f;
		return *this;
	}

	inline Vector3D& operator /= (float f)
	{
		_x /= f;
		_y /= f;
		_z /= f;
		return *this;
	}

	inline float dot (const Vector3D& v)
	{
		return _x * v.x () + _y * v.y () + _z * v.z ();
	}

	inline Vector3D cross (const Vector3D& v)
	{
		return Vector3D (_y * v.z () - _z * v.y (), _z * v.x () - _x * v.z (),
										 _x * v.y () - _y * v.x ());
	}

	inline void setX (float x)
	{
		_x = x;
	}
	inline void setY (float y)
	{
		_y = y;
	}
	inline void setZ (float z)
	{
		_z = z;
	}

private:
	float _x;
	float _y;
	float _z;

};

inline Vector3D operator * (const Vector3D& v, float f)
{
	return Vector3D (f * v.x (), f * v.y (), f * v.z ());
}

inline Vector3D operator * (float f, const Vector3D& v)
{
	return Vector3D (f * v.x (), f * v.y (), f * v.z ());
}

inline Vector3D operator + (const Vector3D& v1, const Vector3D& v2)
{
	return Vector3D (v1.x () + v2.x (), v1.y () + v2.y (), v1.z () + v2.z ());
}

inline Vector3D operator - (const Vector3D& v1, const Vector3D& v2)
{
	return Vector3D (v1.x () - v2.x (), v1.y () - v2.y (), v1.z () - v2.z ());
}

typedef Vector3D Point;

#endif /* VECTOR3D_H_ */
