#pragma once

#include <ncltech\Object.h>

namespace Utils
{

	//Generates bright colour range from 0-1 (based on HSV)
	Vector4 GenColour(float scalar, float alpha);


	Object* BuildSphereObject(const std::string& name, const Vector3& pos, float radius, float invmass = 0.0f, bool collidable = true, bool dragable = true, const Vector4& color = Vector4(1.0f, 1.0f, 1.0f, 1.0f));

	Object* BuildCuboidObject(const std::string& name, const Vector3& pos, const Vector3& scale, float invmass = 0.0f, bool collidable = true, bool dragable = true, const Vector4& color = Vector4(1.0f, 1.0f, 1.0f, 1.0f));
};