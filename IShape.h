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

class IShape
{
public:
	IShape(Point position, Color color, float reflection = 0.0f,
			float transparency = 0.0f, float refracIndex = 0.0f, float spec = 0.0f,
			float diff = 0.0f)
			: _position(position), _color(color), _reflection(reflection), _transparency(
					transparency), _refracIndex(refracIndex), _specularCoefficient(spec), _diffuseCoefficient(
					diff)
	{
	}

	virtual ~IShape()
	{
	}

	inline Point position() const
	{
		return _position;
	}
	inline Color color() const
	{
		return _color;
	}
	inline float reflection() const
	{
		return _reflection;
	}
	inline float transparency() const
	{
		return _transparency;
	}
	inline float specular() const
	{
		return _specularCoefficient;
	}
	inline float diffuse() const
	{
		return _diffuseCoefficient;
	}
	inline float rIndex () const
	{
		return _refracIndex;
	}

	virtual bool intersect(const Ray &, float *, Vector3D&, Color &) = 0;

protected:
	Point _position;
	Color _color;

	float _reflection;
	float _transparency;

	float _diffuseCoefficient;
	float _specularCoefficient;
	float _refracIndex;
};

#endif /* ISHAPE_H_ */
