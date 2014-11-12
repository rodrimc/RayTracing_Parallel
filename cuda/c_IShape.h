/*
 * c_IShape.h
 *
 *  Created on: Oct 31, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef C_ISHAPE_H_
#define C_ISHAPE_H_

#include "c_Vector3D.h"
#include "c_Ray.h"
#include "c_Color.h"

class IShape
{
public:
	__host__ __device__
  IShape(Point position, Color color, float reflection = 0.0f,
			float spec = 0.0f, float diff = 0.0f)
			: _position(position), _color(color), _reflection(reflection),
			  _specularCoefficient(spec), _diffuseCoefficient(
					diff)
	{
	}

  virtual __host__ __device__ ~IShape()
	{
	}

	inline __host__ __device__ Point position() const
	{
		return _position;
	}
	inline __host__ __device__ Color color() const
	{
		return _color;
	}
	inline __host__ __device__ float reflection() const
	{
		return _reflection;
	}
	inline __host__ __device__ float specular() const
	{
		return _specularCoefficient;
	}
	inline __host__ __device__ float diffuse() const
	{
		return _diffuseCoefficient;
	}

	virtual __device__ bool intersect(const Ray &, float *, Vector3D&, Color &) = 0;

protected:
	Point _position;
	Color _color;

	float _reflection;
	float _diffuseCoefficient;
	float _specularCoefficient;
};

#endif /* C_ISHAPE_H_ */

