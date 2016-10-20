#include "PhysicsObject.h"
#include "PhysicsEngine.h"

PhysicsObject::PhysicsObject()
{
	//Initialise Defaults
	m_wsTransformInvalidated = true;
	m_Enabled	= false;

	m_Position = Vector3(0.0f, 0.0f, 0.0f);
	m_LinearVelocity = Vector3(0.0f, 0.0f, 0.0f);
	m_Force = Vector3(0.0f, 0.0f, 0.0f);
	m_InvMass = 0.0f;

	m_Orientation = Quaternion(0.0f, 0.0f, 0.0f, 1.0f);
	m_AngularVelocity = Vector3(0.0f, 0.0f, 0.0f);
	m_Torque = Vector3(0.0f, 0.0f, 0.0f);
	m_InvInertia.ToZero();

	m_colShape = NULL;

	m_Friction = 0.5f;
	m_Elasticity = 0.9f;

	m_OnCollisionCallback = [](PhysicsObject* colliding_obj){ return true; };
}

PhysicsObject::~PhysicsObject()
{
	//Delete ColShape
	if (m_colShape != NULL)
	{
		delete m_colShape;
		m_colShape = NULL;
	}
}

const Matrix4& PhysicsObject::GetWorldSpaceTransform() const 
{
	if (m_wsTransformInvalidated)
	{
		m_wsTransform = m_Orientation.ToMatrix4();
		m_wsTransform.SetPositionVector(m_Position);

		m_wsTransformInvalidated = false;
	}

	return m_wsTransform;
}