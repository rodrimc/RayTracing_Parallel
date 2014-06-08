/*
 * IShape.h
 *
 *  Created on: Jun 5, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef ISHAPE_H_
#define ISHAPE_H_

#include "Vector3D.h"
#include "Ray.h"
#include "Color.h"

class IShape;

struct Intersection
{
	Ray ray;
	float t;
	IShape *pShape;
	Color color;
	Color emitted;
	Vector3D normal;

	Intersection ()
			: ray (), t (far), pShape (0), color (), emitted (), normal ()
	{

	}

	Intersection (const Intersection& i)
			: ray (i.ray), t (i.t), pShape (i.pShape), color (i.color),
				emitted (i.emitted), normal (i.normal)
	{

	}

	Intersection (const Ray& ray)
			: ray (ray), t (ray.farPlane ()), pShape (0), color (), emitted (),
				normal ()
	{

	}

	Intersection& operator = (const Intersection& i)
	{
		ray = i.ray;
		t = i.t;
		pShape = i.pShape;
		color = i.color;
		emitted = i.emitted;
		normal = i.normal;
		return *this;
	}

	bool intersected () const
	{
		return (pShape == 0) ? false : true;
	}

	Point position () const
	{
		return ray.calculate (t);
	}
};

class IShape
{
public:
	IShape (Point position, Color color, float reflection = 0.0f)
			: _position (position), _color (color), _reflection(reflection)
	{
	}

	virtual ~IShape ()
	{
	}

	inline Point position () const { return _position; }
	inline Color color () const { return _color; }
	inline float reflection () const { return _reflection > 0.0f; }

	virtual bool intersect (const Ray &, float *, Vector3D&, Color &) = 0;

protected:
	Point _position;
	Color _color;

	float _reflection;
};

#endif /* ISHAPE_H_ */
