
#pragma once

#include <ncltech\Scene.h>
#include <ncltech\SceneRenderer.h>
#include <ncltech\DistanceConstraint.h>
#include <ncltech\ObjectMesh.h>
#include "Utils.h"

class PhysicsScene2 : public Scene
{
public:
	PhysicsScene2(const std::string& friendly_name)
		: Scene(friendly_name)
	{}

	virtual void OnInitializeScene() override
	{
		SceneManager::Instance()->GetCamera()->SetPosition(Vector3(-3.0f, 10.0f, 15.0f));
		SceneManager::Instance()->GetCamera()->SetYaw(-10.f);
		SceneManager::Instance()->GetCamera()->SetPitch(-30.f);
		PhysicsEngine::Instance()->SetDebugDrawFlags(DEBUHDRAW_FLAGS_CONSTRAINT);

		//Create Ground
		this->AddGameObject(Utils::BuildCuboidObject("Ground", Vector3(0.0f, -1.0f, 0.0f), Vector3(20.0f, 1.0f, 20.0f), 0.0f, true, false, Vector4(0.2f, 0.5f, 1.0f, 1.0f)));

		//Create Pen Walls
		this->AddGameObject(Utils::BuildCuboidObject("Wall1", Vector3(-2.85f, 0.1f, 0.0f), Vector3(0.5f, 0.1f, 3.35f), 0.0f, true, false, Vector4(1.0f, 0.5f, 1.0f, 1.0f)));
		this->AddGameObject(Utils::BuildCuboidObject("Wall2", Vector3(2.85f, 0.1f, 0.0f), Vector3(0.5f, 0.1f, 3.35f), 0.0f, true, false, Vector4(1.0f, 0.5f, 1.0f, 1.0f)));
		this->AddGameObject(Utils::BuildCuboidObject("Wall3", Vector3(0.0f, 0.1f, -2.85f), Vector3(2.35f, 0.1f, 0.5f), 0.0f, true, false, Vector4(1.0f, 0.5f, 1.0f, 1.0f)));
		this->AddGameObject(Utils::BuildCuboidObject("Wall4", Vector3(0.0f, 0.1f, 2.85f), Vector3(2.35f, 0.1f, 0.5f), 0.0f, true, false, Vector4(1.0f, 0.5f, 1.0f, 1.0f)));

		//Create Sphere Triangle Stack
		for (int y = 0; y < 5; ++y)
		{
			Vector4 colour = Utils::GenColour(y * 0.2f, 1.0f);
			for (int x = 0; x <= y; ++x)
			{
				for (int z = 0; z <= y; ++z)
				{
					Object* sphere = Utils::BuildSphereObject("", Vector3(x - y * 0.5f, 3.6f - y * 0.75f, z - y * 0.5f), 0.5f, 5.f, true, true, colour);
					sphere->Physics()->SetFriction(1.0f);
					this->AddGameObject(sphere);
				}
			}
		}

		//Wrecking Ball

		//Build Chain
		PhysicsObject* lastObject = NULL;
		for (int x = 0; x < 5; ++x)
		{
			PhysicsObject* tmp = new PhysicsObject();
			tmp->SetPosition(Vector3(0.0f, 23.0f, -x * 4.0f));
			tmp->SetInverseMass(0.0f);
			tmp->SetInverseInertia(Matrix3::ZeroMatrix);
			if (lastObject != NULL)
			{
				tmp->SetInverseMass(1.0f);

				PhysicsEngine::Instance()->AddConstraint(new DistanceConstraint(
					tmp,
					lastObject,
					tmp->GetPosition(),
					lastObject->GetPosition()
					));
			}
			lastObject = tmp;
			PhysicsEngine::Instance()->AddPhysicsObject(tmp);
		}

		Object* ball = Utils::BuildSphereObject("", Vector3(0.0f, 23.0f, -20.0f), 1.5f, 0.1f, true, false, Vector4(0.0f, 0.0f, 0.0f, 1.0f));	
		this->AddGameObject(ball);

		DistanceConstraint* dc = new DistanceConstraint(
			lastObject,
			ball->Physics(),
			lastObject->GetPosition(),
			ball->Physics()->GetPosition() + Vector3(0.0f, 0.0f, 1.5f)
			);
		PhysicsEngine::Instance()->AddConstraint(dc);
		
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