#include "CollidableObject.h"
#include <ncltech/ScreenPicker.h>
#include <ncltech/NCLDebug.h>
#include <libsim/XPBD.h>

CollidableObject::CollidableObject(const std::string& name, XPBDSphereConstraint* target)
	: ObjectMesh(name)
	, m_Target(target)
	, m_LocalClickOffset(0.0f, 0.0f, 0.0f)
	, m_MouseDownColOffset(0.2f, 0.2f, 0.2f, 0.0f)
	, m_MouseOverColOffset(0.2f, 0.2f, 0.2f, 0.0f)
{
	//Register the object to listen for click callbacks
	ScreenPicker::Instance()->RegisterObject(this);
}

CollidableObject::~CollidableObject()
{
	//Unregister the object to prevent sending click events to undefined memory
	ScreenPicker::Instance()->UnregisterObject(this);
}

void CollidableObject::SetMouseOverColourOffset(const Vector4& col_offset)
{
	m_MouseOverColOffset = col_offset;
}

void CollidableObject::SetMouseDownColourOffset(const Vector4& col_offset)
{
	m_MouseOverColOffset = m_MouseDownColOffset - col_offset;
}

void CollidableObject::OnMouseEnter(float dt)
{
	this->m_Colour += m_MouseOverColOffset;
}

void CollidableObject::OnMouseLeave(float dt)
{
	this->m_Colour -= m_MouseOverColOffset;
}

void CollidableObject::OnMouseDown(float dt, const Vector3& worldPos)
{
	m_LocalClickOffset = worldPos - this->m_WorldTransform.GetPositionVector();
	this->m_Colour += m_MouseDownColOffset;

	if (this->HasPhysics())
	{
		this->Physics()->SetAngularVelocity(Vector3(0.0f, 0.0f, 0.0f));
		this->Physics()->SetLinearVelocity(Vector3(0.0f, 0.0f, 0.0f));
	}
}

void CollidableObject::OnMouseMove(float dt, const Vector3& worldPos, const Vector3& worldChange)
{
	Vector3 newpos = worldPos - m_LocalClickOffset;

	if (this->HasPhysics())
	{
		this->Physics()->SetPosition(worldPos - m_LocalClickOffset);
		this->Physics()->SetAngularVelocity(Vector3(0.0f, 0.0f, 0.0f));
		this->Physics()->SetLinearVelocity(worldChange / dt * 0.5f);
	}
	else
	{
		this->m_LocalTransform.SetPositionVector(worldPos - m_LocalClickOffset);
		m_Target->centre.x = worldPos.x - m_LocalClickOffset.x;
		m_Target->centre.y = worldPos.y - m_LocalClickOffset.y;
		m_Target->centre.z = worldPos.z - m_LocalClickOffset.z;
	}
}

void CollidableObject::OnMouseUp(float dt, const Vector3& worldPos)
{
	if (this->HasPhysics())
	{
		this->Physics()->SetPosition(worldPos - m_LocalClickOffset);
	}
	else
	{
		this->m_LocalTransform.SetPositionVector(worldPos - m_LocalClickOffset);

		m_Target->centre.x = worldPos.x - m_LocalClickOffset.x;
		m_Target->centre.y = worldPos.y - m_LocalClickOffset.y;
		m_Target->centre.z = worldPos.z - m_LocalClickOffset.z;
	}

	this->m_Colour -= m_MouseDownColOffset;
}