/*
 * c_Color.h
 *
 *  Created on: Oct 31, 2014
 *      Author: Rodrigo Costa
 *			e-mail: rodrigocosta@telemidia.puc-rio.br
 */

#ifndef C_COLOR_H_
#define C_COLOR_H_

#include <algorithm>

class Color
{
public:
	__host__ __device__ Color ()
			: _r (0.0f), _g (0.0f), _b (0.0f)
	{
	}
	__host__ __device__ Color (const Color& c)
			: _r (c.r ()), _g (c.g ()), _b (c.b ())
	{
	}
	__host__ __device__ Color (float r, float g, float b)
			: _r (r), _g (g), _b (b)
	{
	}
	explicit __host__ __device__ Color (float f)
			: _r (f), _g (f), _b (f)
	{
	}

	inline void __device__ clamp (float min = 0.0f, float max = 1.0f)
	{
		_r = std::max (min, std::min (max, _r));
		_g = std::max (min, std::min (max, _g));
		_b = std::max (min, std::min (max, _b));
	}

	inline __host__ __device__ float r () const
	{
		return _r;
	}
	inline __host__ __device__ float g () const
	{
		return _g;
	}
	inline __host__ __device__ float b () const
	{
		return _b;
	}

	inline __host__ __device__ Color& operator = (const Color& c)
	{
		_r = c.r ();
		_g = c.g ();
		_b = c.b ();
		return *this;
	}

	inline __device__ Color& operator += (const Color& c)
	{
		_r += c.r ();
		_g += c.g ();
		_b += c.b ();
		return *this;
	}

	inline __device__ Color& operator -= (const Color& c)
	{
		_r -= c.r ();
		_g -= c.g ();
		_b -= c.b ();
		return *this;
	}

	inline __device__ Color& operator *= (const Color& c)
	{
		_r *= c.r ();
		_g *= c.g ();
		_b *= c.b ();
		return *this;
	}

	inline __device__ Color& operator /= (const Color& c)
	{
		_r /= c.r ();
		_g /= c.g ();
		_b /= c.b ();
		return *this;
	}

	inline __device__ Color& operator *= (float f)
	{
		_r *= f;
		_g *= f;
		_b *= f;
		return *this;
	}

	inline __device__ Color& operator /= (float f)
	{
		_r /= f;
		_g /= f;
		_b /= f;
		return *this;
	}

private:
	float _r;
	float _g;
	float _b;
};

inline __device__ Color operator * (const Color& c, float f)
{
	return Color (f * c.r (), f * c.g (), f * c.b ());
}

inline __device__ Color operator * (float f, const Color& c)
{
	return Color (f * c.r (), f * c.g (), f * c.b ());
}

inline __device__ Color operator * (const Color& c1, const Color& c2)
{
	return Color (c1.r () * c2.r (), c1.g () * c2.g (), c1.b () * c2.b ());
}

inline __device__ Color operator +(const Color& c1, const Color& c2)
{
    return Color(c1.r() + c2.r(),
                 c1.g() + c2.g(),
                 c1.b() + c2.b());
}

inline __device__ Color operator +(const Color& c, const float& f)
{
    return Color(c.r() + f,
                 c.g() + f,
                 c.b() + f);
}

#endif /* C_COLOR_H_ */
