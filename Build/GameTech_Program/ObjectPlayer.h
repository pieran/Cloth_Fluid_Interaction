
#pragma once
#include <ncltech\ObjectMesh.h>
#include <ncltech\NCLDebug.h>

class ObjectPlayer : public ObjectMesh
{
public:
	ObjectPlayer(const std::string& name) : ObjectMesh(name) {}

protected:
	virtual void	OnUpdateObject(float dt) override
	{
		if (!this->HasPhysics())
		{
			NCLERROR("Player object does not have a valid physics node to manipulate!");
		}
		else
		{
			const float mv_speed = 5.f * dt;			//Meters per second
			const float rot_speed = 90.f * dt;			//Degrees per second

			if (Window::GetKeyboard()->KeyDown(KEYBOARD_UP))
			{
				Physics()->SetPosition(Physics()->GetPosition() +
					Physics()->GetOrientation().ToMatrix3() * Vector3(0.0f, 0.0f, -mv_speed));
			}

			if (Window::GetKeyboard()->KeyDown(KEYBOARD_DOWN))
			{
				Physics()->SetPosition(Physics()->GetPosition() +
					Physics()->GetOrientation().ToMatrix3()* Vector3(0.0f, 0.0f, mv_speed));
			}

			if (Window::GetKeyboard()->KeyDown(KEYBOARD_LEFT))
			{
				Physics()->SetOrientation(Physics()->GetOrientation() *
					Quaternion::AxisAngleToQuaterion(Vector3(0.0f, 1.0f, 0.0f), rot_speed));
			}

			if (Window::GetKeyboard()->KeyDown(KEYBOARD_RIGHT))
			{
				Physics()->SetOrientation(Physics()->GetOrientation() *
					Quaternion::AxisAngleToQuaterion(Vector3(0.0f, 1.0f, 0.0f), -rot_speed));
			}
		}
	}
};