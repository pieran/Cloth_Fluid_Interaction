

#pragma once

#include "Constraint.h"
#include "NCLDebug.h"
#include "PhysicsEngine.h"

class DistanceConstraint : public Constraint
{
public:
	DistanceConstraint(PhysicsObject* objA, PhysicsObject* objB,
		const Vector3& globalOnA, const Vector3& globalOnB)
	{
		this->objA = objA;
		this->objB = objB;

		Vector3 ab = globalOnB - globalOnA;
		this->distance = ab.Length();

		Vector3 r1 = (globalOnA - objA->GetPosition());
		Vector3 r2 = (globalOnB - objB->GetPosition());
		localOnA = Matrix3::Transpose(objA->GetOrientation().ToMatrix3()) * r1;
		localOnB = Matrix3::Transpose(objB->GetOrientation().ToMatrix3()) * r2;
	}

	virtual void ApplyImpulse() override
	{
		if (objA->GetInverseMass() + objB->GetInverseMass() == 0.0f)
			return;

		Vector3 r1 = objA->GetOrientation().ToMatrix3() * localOnA;
		Vector3 r2 = objB->GetOrientation().ToMatrix3() * localOnB;

		Vector3 globalOnA = r1 + objA->GetPosition();
		Vector3 globalOnB = r2 + objB->GetPosition();

		Vector3 ab = globalOnB - globalOnA;
		Vector3 abn = ab;
		abn.Normalise();



		Vector3 v0 = objA->GetLinearVelocity() + Vector3::Cross(objA->GetAngularVelocity(), r1);
		Vector3 v1 = objB->GetLinearVelocity() + Vector3::Cross(objB->GetAngularVelocity(), r2);
		{
			float constraintMass = (objA->GetInverseMass() + objB->GetInverseMass()) +
				Vector3::Dot(abn,
				Vector3::Cross(objA->GetInverseInertia()*Vector3::Cross(r1, abn), r1) +
				Vector3::Cross(objB->GetInverseInertia()*Vector3::Cross(r2, abn), r2));

			//Baumgarte Offset (Adds energy to the system to counter slight solving errors that accumulate over time - known as 'constraint drift')
			float b = 0.0f;
			{
				float distance_offset = ab.Length() - distance;
				float baumgarte_scalar = 0.1f;
				b = -(baumgarte_scalar / PhysicsEngine::Instance()->GetDeltaTime()) * distance_offset;
			}

			float jn = -(Vector3::Dot(v0 - v1, abn) + b) / constraintMass;

			objA->SetLinearVelocity(objA->GetLinearVelocity() + abn*(jn*objA->GetInverseMass()));
			objB->SetLinearVelocity(objB->GetLinearVelocity() - abn*(jn*objB->GetInverseMass()));

			objA->SetAngularVelocity(objA->GetAngularVelocity() + objA->GetInverseInertia()* Vector3::Cross(r1, abn * jn));
			objB->SetAngularVelocity(objB->GetAngularVelocity() - objB->GetInverseInertia()* Vector3::Cross(r2, abn * jn));
		}


	}

	virtual void DebugDraw() const
	{
		Vector3 globalOnA = objA->GetOrientation().ToMatrix3() * localOnA + objA->GetPosition();
		Vector3 globalOnB = objB->GetOrientation().ToMatrix3() * localOnB + objB->GetPosition();

		NCLDebug::DrawThickLine(globalOnA, globalOnB, 0.02f, Vector4(0.0f, 0.0f, 0.0f, 1.0f));
		NCLDebug::DrawPointNDT(globalOnA, 0.05f, Vector4(1.0f, 0.8f, 1.0f, 1.0f));
		NCLDebug::DrawPointNDT(globalOnB, 0.05f, Vector4(1.0f, 0.8f, 1.0f, 1.0f));
	}

protected:
	PhysicsObject *objA, *objB;
	float   distance;
	Vector3 localOnA, localOnB;
};