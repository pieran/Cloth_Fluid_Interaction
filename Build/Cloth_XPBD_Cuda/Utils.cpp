#include "utils.h"
#include <ncltech\ObjectMesh.h>
#include <ncltech\ObjectMeshDragable.h>
#include <ncltech\SphereCollisionShape.h>
#include <ncltech\CuboidCollisionShape.h>
#include <ncltech\CommonMeshes.h>

Vector4 Utils::GenColour(float scalar, float alpha)
{
	Vector4 c;
	c.w = alpha;

	float t;
	c.x = abs(modf(scalar + 1.0f, &t) * 6.0f - 3.0f) - 1.0f;
	c.y = abs(modf(scalar + 2.0f / 3.0f, &t) * 6.0f - 3.0f) - 1.0f;
	c.z = abs(modf(scalar + 1.0f / 3.0f, &t) * 6.0f - 3.0f) - 1.0f;

	c.x = min(max(c.x, 0.0f), 1.0f);
	c.y = min(max(c.y, 0.0f), 1.0f);
	c.z = min(max(c.z, 0.0f), 1.0f);

	return c;
}

Object* Utils::BuildSphereObject(const std::string& name, const Vector3& pos, float radius, float invmass, bool collidable, bool dragable, const Vector4& color)
{
	ObjectMesh* sphere = dragable
		? new ObjectMeshDragable(name)
		: new ObjectMesh(name);

	sphere->SetMesh(CommonMeshes::Sphere(), false);
	sphere->SetTexture(CommonMeshes::CheckerboardTex(), false);
	sphere->SetLocalTransform(Matrix4::Scale(Vector3(radius, radius, radius)));
	sphere->SetColour(color);
	sphere->SetBoundingRadius(radius);

	if (collidable)
	{
		sphere->CreatePhysicsNode();

		sphere->Physics()->SetPosition(pos);
		sphere->Physics()->SetCollisionShape(new SphereCollisionShape(radius));

		sphere->Physics()->SetInverseMass(invmass);
		sphere->Physics()->SetInverseInertia(sphere->Physics()->GetCollisionShape()->BuildInverseInertia(sphere->Physics()->GetInverseMass()));
	}
	else
	{
		sphere->SetLocalTransform(Matrix4::Translation(pos) * sphere->GetLocalTransform());
	}

	return sphere;
}

Object* Utils::BuildCuboidObject(const std::string& name, const Vector3& pos, const Vector3& halfdims, float invmass, bool collidable, bool dragable, const Vector4& color)
{
	ObjectMesh* cuboid = dragable
		? new ObjectMeshDragable(name)
		: new ObjectMesh(name);

	cuboid->SetMesh(CommonMeshes::Cube(), false);
	cuboid->SetTexture(CommonMeshes::CheckerboardTex(), false);
	cuboid->SetLocalTransform(Matrix4::Scale(halfdims));
	cuboid->SetColour(color);
	cuboid->SetBoundingRadius(halfdims.Length());

	if (collidable)
	{
		cuboid->CreatePhysicsNode();
		cuboid->Physics()->SetPosition(pos);

		cuboid->Physics()->SetCollisionShape(new CuboidCollisionShape(halfdims));

		cuboid->Physics()->SetInverseMass(invmass);
		cuboid->Physics()->SetInverseInertia(cuboid->Physics()->GetCollisionShape()->BuildInverseInertia(cuboid->Physics()->GetInverseMass()));
	}
	else
	{
		cuboid->SetLocalTransform(Matrix4::Translation(pos) * cuboid->GetLocalTransform());
	}

	return cuboid;
}
