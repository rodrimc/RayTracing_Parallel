#include "util.h"
#define MAX_DEPTH 7

void writePPMFile(Image *image, const char *filename, float width, float height)
{
	std::ofstream ofs(filename, std::ios::out | std::ios::binary);
	ofs << "P6\n" << width << " " << height << "\n255\n";
	for (unsigned i = 0; i < width * height; ++i)
	{
		Image pixel = image[i];

		ofs << (unsigned char) (std::min(float(1), pixel.r()) * 255)
				<< (unsigned char) (std::min(float(1), pixel.g()) * 255)
				<< (unsigned char) (std::min(float(1), pixel.b()) * 255);
	}
	ofs.close();
}

void initScene(std::set<IShape *>&sceneShapes, std::set<Light *>&sceneLights)
{
	Plane *mirror = new Plane (Point(0.0f, 0.0, -15.0f), Vector3D (0.0f, 0.0f, 1.0f), 
      Color(0.2f, 0.2f, 0.2f), 1.0f, 0.2f, 0.2f, false);

	Plane *floor = new Plane(Point(0.0f, -2.5f, 0.0f), Vector3D(0.0f, 1.0f, 0.0f),
			Color(1.0f, 1.0f, 1.0f), 0.5f, 0.3f, 0.3f);

	Sphere *sphere0 = new Sphere(Point(-5.0f, 1.0f, 0.0f), Color(0.5, 0.1, 0.1),
			0.5f, 0.5f, 1.0f, 0.6f);

	Sphere *sphere1 = new Sphere(Point(3.0f, 0.0f, 0.0f), Color(1.0, 0.1, 0.1),
			1.5f, 0.8f, 1.0f, 0.2f);
	Sphere *sphere2 = new Sphere(Point(-3.0f, 0.0f, 0.0f), Color(0.1, 1.0, 0.1),
			1.5f, 0.5f, 1.0f, 0.7f);
	Sphere *sphere3 = new Sphere(Point(0.0f, 0.0f, -4.0f), Color(0.5, 0.5, 0.5),
			1.5f, 0.5f, 1.0f, 0.7f);

	Sphere *sphere4 = new Sphere(Point(10.0f, 5.0f, -4.0f), Color(1.0, 0.3, 0.8),
				1.5f, 0.8f, 1.0f, 0.4f);

	Sphere *sphere5 = new Sphere(Point(8.0f, 0.0f, 4.0f), Color(0.5, 0.5, 1.0),
				1.5f, 0.3f, 1.0f, 0.7f);

	Sphere *sphere6 = new Sphere(Point(5.0f, 10.0f, 0.0f), Color(0.3, 0.6, 0.1),
				1.5f, 1.0f, 1.0f, 0.7f);

	Sphere *sphere7 = new Sphere(Point(-3.0f, 4.0f, -2.0f), Color(0.1, 0.6, 0.7),
				1.5f, 0.2f, 1.0f, 0.7f);

	Sphere *sphere8 = new Sphere(Point(-4.0f, 7.0f, 3.0f), Color(0.5, 0.1, 0.7),
				1.5f, 0.8f, 1.0f, 0.7f);


	sceneShapes.insert(floor);
  sceneShapes.insert(mirror);

  sceneShapes.insert(sphere0);
	sceneShapes.insert(sphere1);
	sceneShapes.insert(sphere2);
	sceneShapes.insert(sphere3);
	sceneShapes.insert(sphere4);
	sceneShapes.insert(sphere5);
	sceneShapes.insert(sphere6);
	sceneShapes.insert(sphere7);
	sceneShapes.insert(sphere8);

	Light *frontLight = new Light(Point(0.0f, 13.0f, 10.0f),
				Color(1.0f, 1.0f, 1.0f), 1.0f);

	sceneLights.insert(frontLight);
}

IShape* calculateIntersect (const Ray &ray, std::set<IShape*> &sceneShapes,
														float *t, Vector3D &shapeNormal, Color &pixelColor)
{
	*t = INFINITY;
	Color color;
	Vector3D normal;
	IShape *shapeIntersection = 0;
	for (auto shape : sceneShapes)
	{
		float near;
		if (shape->intersect (ray, &near, normal, color) && near < *t)
		{
			shapeNormal = normal;
			pixelColor = color;
			*t = near;
			shapeIntersection = shape;
		}
	}
	return shapeIntersection;
}

Color ambientColor (const Color& color)
{
	return ambient_coefficient * color;
}

Color specularColor (const Vector3D &direction, const Vector3D &normal,
										 const Ray& ray, const Light* light,
										 const float &specularCoefficient)
{
	Color specularColor;
	Vector3D refl = direction - normal * 2 * Vector3D (direction).dot (normal);
	refl.normalize ();

	float m = refl.dot (ray.direction ());
	m = m < 0 ? 0.0f : m;

	float cosB = m / refl.length () * ray.direction ().length ();
	float spec = pow (cosB, 50);

	specularColor = specularCoefficient * light->color () * spec;

	return specularColor;
}

Color diffuseColor (const Vector3D& direction, const Light *light,
										const Vector3D& normal, const Color& color,
										const float &diffuseCoefficient)
{
	Color diffuseColor;
	float dot = Vector3D (normal).dot (direction);
	dot = dot < 0 ? 0.0f : dot;

	diffuseColor = diffuseCoefficient * color * light->color () * dot;

	return diffuseColor;
}

Color trace (const Ray& ray, std::set<IShape*>& sceneShapes,
						 std::set<Light*>& sceneLights, int depth)
{
	Color pixelColor (0.3);

	float near;
	Color color;
	Vector3D normal;
	IShape *shape = calculateIntersect (ray, sceneShapes, &near, normal, color);
	if (shape)
	{
		Point intersectionPoint = ray.calculate (near);

		Vector3D n;
		Color c;

		pixelColor = Color (0.0f);

		//Calculate illumination on intersected pixel
		for (auto light : sceneLights)
		{
			Vector3D lightDirection = (light->position () - intersectionPoint);

			float lightLenght = lightDirection.normalize ();

			const Ray shadowRay (intersectionPoint + normal * bias, lightDirection,
													 lightLenght);
			near = INFINITY;

			IShape *s = calculateIntersect (shadowRay, sceneShapes, &near, n, c);
			if (!s) //There is no object between the intersected pixel and this light.
			{
				float diffuseCoefficient = shape->diffuse ();
				float specularCoefficient = shape->specular ();

				pixelColor += ambientColor (color);
				if (diffuseCoefficient > 0.0f)
					pixelColor += diffuseColor (lightDirection, light, normal, color,
																			diffuseCoefficient);

				if (specularCoefficient > 0.0f)
					pixelColor += specularColor (lightDirection, normal, ray, light,
																			 specularCoefficient);
			}
			else //Intersected pixel is shadowed!!!
			{
				pixelColor = color * 0.1;
				break;
			}
		}

		//Calculate the reflected color
		if ((shape->reflection () > 0)
				&& depth <= MAX_DEPTH)
		{
			Vector3D reflDir = ray.direction ()
					- normal * 2 * ray.direction ().dot (normal);
			reflDir.normalize ();

			Ray reflectionRay (intersectionPoint + normal * bias, reflDir);
			Color reflectionColor = trace (reflectionRay, sceneShapes, sceneLights,
			                               depth + 1);

			pixelColor += reflectionColor * shape->reflection ();
		}
	}

	pixelColor.clamp ();
	return pixelColor;
}
