#include "Plane.h"

Plane::Plane(const Vector3 &normal, float distance, bool normalise) {
	if(normalise) {
		float length = normal.Length();

		this->normal   = normal		/ length;
		this->distance = distance	/ length;
	}
	else{
		this->normal = normal;
		this->distance = distance;
	}
}

bool Plane::SphereInPlane(const Vector3 &position, float radius) const {
	if(Vector3::Dot(position,normal)+distance <= -radius) {
		return false;
	}

	return true;	
}

bool Plane::PointInPlane(const Vector3 &position) const {
	float test = Vector3::Dot(position,normal);
	float test2 = test + distance;

	if(Vector3::Dot(position,normal)+distance < -0.0001f) {
		return false;
	}

	return true;
}