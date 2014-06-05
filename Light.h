/*
 * Light.h
 *
 *  Created on: Jun 5, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef LIGHT_H_
#define LIGHT_H_

#include "Color.h"
#include "Vector3D.h"

class Light
{
public:
	Light (Vector3D position, Color color, float power)
			: _position (position), _color (color), _power(power)
	{
	}

	Vector3D position () const
	{
		return _position;
	}

	Color color () const
	{
		return _color * _power;
	}

protected:
	Vector3D _position;
	Color _color;
	float _power;
};

class RectangleLight : public Light
{
public:
	RectangleLight (const Point& pos, const Vector3D& side1,
									const Vector3D& side2, const Color& color, float power)
			: Light (pos, color, power), _side1 (side1), _side2 (side2)
	{
	}

	virtual ~RectangleLight ()
	{
	}

	virtual bool intersect (Intersection& intersection)
	{
		// This is much like a plane intersection, except we also range check it
		// to make sure it's within the rectangle.  Please see the plane shape
		// intersection method for a little more info.

		Vector3D normal = _side1.cross (_side2).normalized ();
		float nDotD = normal.dot (intersection.ray.direction ());
		if (nDotD == 0.0f)
		{
			return false;
		}

		float t = (_position.dot (normal) - intersection.ray.origin ().dot (normal))
				/ intersection.ray.direction ().dot (normal);

		// Make sure t is not behind the ray, and is closer than the current
		// closest intersection.
		if (t >= intersection.t || t < kRayTMin)
		{
			return false;
		}

		// Take the intersection point on the plane and transform it to a local
		// space where we can really easily check if it's in range or not.
		Vector3D side1Norm = _side1;
		Vector3D side2Norm = _side2;
		float side1Length = side1Norm.normalize ();
		float side2Length = side2Norm.normalize ();

		Point worldPoint = intersection.ray.calculate (t);
		Point worldRelativePoint = worldPoint - _position;
		Point localPoint = Point (worldRelativePoint.dot (side1Norm),
															worldRelativePoint.dot (side2Norm), 0.0f);

		// Do the actual range check
		if (localPoint.x () < 0.0f || localPoint.x () > side1Length
				|| localPoint.y () < 0.0f || localPoint.y () > side2Length)
		{
			return false;
		}

		// This intersection is the closest so far, so record it.
		intersection.t = t;
//		intersection.pShape = this;
		intersection.color = Color ();
		intersection.emitted = this->color ();
		intersection.normal = normal;
		// Hit the back side of the light?  We'll count it, so flip the normal
		// to effectively make a double-sided light.
		if (intersection.normal.dot (intersection.ray.direction ()) > 0.0f)
		{
			intersection.normal *= -1.0f;
		}

		return true;
	}

	// Given two random numbers between 0.0 and 1.0, find a location + surface
	// normal on the surface of the *light*.
	virtual bool sampleSurface (float u1, float u2,
															const Point& referencePosition,
															Point& outPosition, Vector3D& outNormal)
	{
		outNormal = _side1.cross (_side2).normalized ();
		outPosition = _position + _side1 * u1 + _side2 * u2;
		// Reference point out in back of the light?  That's okay, we'll flip
		// the normal to have a double-sided light.
		if (outNormal.dot (outPosition - referencePosition) > 0.0f)
		{
			outNormal *= -1.0f;
		}
		return true;
	}

protected:
	Vector3D _side1;
	Vector3D _side2;
};

#endif /* LIGHT_H_ */
