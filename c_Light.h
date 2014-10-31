/*
 * c_Light.h
 *
 *  Created on: Oct 31, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef C_LIGHT_H_
#define C_LIGHT_H_

#include "c_Color.h"
#include "c_Vector3D.h"

class Light
{
public:
	Light (Vector3D position, Color color, float power, float diffuse = 0.3f)
			: _position (position), _color (color), _power(power), _spec(diffuse)
	{
	}

	__device__ Vector3D position () const
	{
		return _position;
	}

	__device__ Color color () const
	{
		return _color * _power;
	}

	__device__ float power () const { return _power; }

	__device__ float specular () const { return _spec; }

protected:
	Vector3D _position;
	Color _color;
	float _power;
	float _spec;
	};

#endif /* C_LIGHT_H_ */

