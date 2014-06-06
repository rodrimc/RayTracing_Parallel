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

#endif /* LIGHT_H_ */
