
#pragma once

#include <ncltech\Scene.h>
#include <ncltech\SceneRenderer.h>
#include <ncltech\DistanceConstraint.h>
#include "Utils.h"

class PhysicsScene3 : public Scene
{
public:
	PhysicsScene3(const std::string& friendly_name)
		: Scene(friendly_name)
	{}

	virtual void OnInitializeScene() override
	{
		SceneManager::Instance()->GetCamera()->SetPosition(Vector3(-3.0f, 10.0f, 15.0f));
		SceneManager::Instance()->GetCamera()->SetYaw(-10.f);
		SceneManager::Instance()->GetCamera()->SetPitch(-20.f);

		PhysicsEngine::Instance()->SetDebugDrawFlags(DEBUHDRAW_FLAGS_CONSTRAINT);

		//Create Ground
		this->AddGameObject(Utils::BuildCuboidObject("Ground", Vector3(0.0f, -1.0f, 0.0f), Vector3(20.0f, 1.0f, 20.0f), 0.0f, true, false, Vector4(0.2f, 0.5f, 1.0f, 1.0f)));

		//Create Wheel
		Object* wheel = Utils::BuildCuboidObject("wheel", Vector3(0.0f, 5.5f, 0.0f), Vector3(3.0f, 0.5f, 3.0f), 1.0f, true, false, Vector4(1.0f, 0.7f, 0.5f, 1.0f));
		wheel->Physics()->SetInverseMass(0.0f); //CANT MOVE

		//CANT ROTATE IN Z/X AXIS
		Matrix3 inertia = wheel->Physics()->GetInverseInertia();
		inertia(0, 0) = 0.0f;
		inertia(2, 2) = 0.0f;
		wheel->Physics()->SetInverseInertia(inertia);

		wheel->Physics()->SetAngularVelocity(Vector3(0.0f, 1.0f, 0.0f));
		this->AddGameObject(wheel);

		//Create Seats
		uint numSeats = 10;
		for (uint i = 0; i < numSeats; ++i)
		{
			float angleX = cos(i / (float)numSeats * PI * 2.0f);
			float angleY = sin(i / (float)numSeats * PI * 2.0f);
			Vector3 pos = wheel->Physics()->GetPosition() + Vector3(angleX, 0.0f, angleY) * 6.0f;

			Object* obj = Utils::BuildCuboidObject("", pos, Vector3(0.5f, 0.5f, 0.5f), 20.0f, true, true, Vector4(0.7f, 1.0f, 0.7f, 1.0f));
			obj->Physics()->SetInverseInertia(Matrix3::ZeroMatrix); //CANT ROTATE
			this->AddGameObject(obj);

			Vector3 normal = wheel->Physics()->GetPosition() - pos;
			obj->Physics()->SetLinearVelocity(Vector3::Cross(normal, wheel->Physics()->GetAngularVelocity()));
			normal.Normalise();


			DistanceConstraint* dc = new DistanceConstraint(
				obj->Physics(),
				wheel->Physics(),
				obj->Physics()->GetPosition(),
				wheel->Physics()->GetPosition() - normal * 3.0f
				);

			PhysicsEngine::Instance()->AddConstraint(dc);
		}
	}

	virtual void OnUpdateScene(float dt) override
	{
		Scene::OnUpdateScene(dt);

		uint drawFlags = PhysicsEngine::Instance()->GetDebugDrawFlags();

		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "Physics:");
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "     Draw Collision Volumes : %s (Press C to toggle)", (drawFlags & DEBUHDRAW_FLAGS_COLLISIONVOLUMES) ? "Enabled" : "Disabled");
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "     Draw Collision Normals : %s (Press N to toggle)", (drawFlags & DEBUHDRAW_FLAGS_COLLISIONNORMALS) ? "Enabled" : "Disabled");
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "     Draw Manifolds : %s (Press M to toggle)", (drawFlags & DEBUHDRAW_FLAGS_MANIFOLD) ? "Enabled" : "Disabled");


		if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_C))
			drawFlags ^= DEBUHDRAW_FLAGS_COLLISIONVOLUMES;

		if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_N))
			drawFlags ^= DEBUHDRAW_FLAGS_COLLISIONNORMALS;

		if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_M))
			drawFlags ^= DEBUHDRAW_FLAGS_MANIFOLD;

		PhysicsEngine::Instance()->SetDebugDrawFlags(drawFlags);
	}
};