/*#include "Constraint.h"

Constraint::Constraint() {
	this->objA = NULL;
	this->objB = NULL;

	this->b = 0.0f;

	delta = 0.0f;
	softness = 0.0f;
	impulseSum = 0.0f;
	impulseSumMin = -FLT_MAX;
	impulseSumMax = FLT_MAX;
}

Constraint::Constraint(PhysicsObject* objA, PhysicsObject* objB,
	Vector3 j1, Vector3 j2, Vector3 j3, Vector3 j4, float b) {
	this->objA = objA;
	this->objB = objB;

	this->j1 = j1;
	this->j2 = j2;
	this->j3 = j3;
	this->j4 = j4;
	this->b = b;

	delta = 0.0f;
	softness = 0.0f;
	impulseSum = 0.0f;
	impulseSumMin = -FLT_MAX;
	impulseSumMax = FLT_MAX;
}

void Constraint::ApplyImpulse()
{
	delta = 0.0f;

	//J * M(-1) * J(t)
	float contstraint_mass = objA->GetInverseMass() * Vector3::Dot(j1, j1)
		+ Vector3::Dot(j2, (objA->GetInverseInertia() * j2))
		+ objB->GetInverseMass() * Vector3::Dot(j3, j3)
		+ Vector3::Dot(j4, (objB->GetInverseInertia() * j4))
		+ softness;

	if (contstraint_mass > 0.00001f)
	{
		//JV
		float jv = Vector3::Dot(j1, objA->GetLinearVelocity())
			+ Vector3::Dot(j2, objA->GetAngularVelocity())
			+ Vector3::Dot(j3, objB->GetLinearVelocity())
			+ Vector3::Dot(j4, objB->GetAngularVelocity());

		float denom = -(jv + b);
		delta = denom / contstraint_mass;

		float oldImpulseSum = impulseSum;
		impulseSum = min(max(impulseSum + delta, impulseSumMin), impulseSumMax);
		float realDelta = impulseSum - oldImpulseSum;

		objA->SetLinearVelocity(objA->GetLinearVelocity() + (j1 * realDelta) * objA->GetInverseMass());
		objA->SetAngularVelocity(objA->GetAngularVelocity() + objA->GetInverseInertia() * (j2 * realDelta));
		objB->SetLinearVelocity(objB->GetLinearVelocity() + (j3 * realDelta) * objB->GetInverseMass());
		objB->SetAngularVelocity(objB->GetAngularVelocity() + objB->GetInverseInertia() * (j4 * realDelta));
	}
}*/