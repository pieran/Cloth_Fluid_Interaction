
#pragma once

#include <ncltech\Scene.h>
#include <ncltech\SceneManager.h>
#include "Utils.h"

class GameTechTutorial4 : public Scene
{
public:
	GameTechTutorial4(const std::string& friendly_name)
		: Scene(friendly_name)
	{}

	virtual void OnInitializeScene() override
	{
		SceneManager::Instance()->GetCamera()->SetPosition(Vector3(-3.0f, 10.0f, 15.0f));
		SceneManager::Instance()->GetCamera()->SetYaw(-10.f);
		SceneManager::Instance()->GetCamera()->SetPitch(-30.f);

		//Create Ground
		this->AddGameObject(Utils::BuildCuboidObject("Ground", Vector3(0.0f, -1.0f, 0.0f), Vector3(20.0f, 1.0f, 20.0f), 0.0f, true, false, Vector4(0.2f, 0.5f, 1.0f, 1.0f)));

		//COLLISION RESOLUTION
		{
			//Stacking Box Pyramid
			for (int y = 0; y < 3; ++y)
			{
				for (int x = 0; x <= y; ++x)
				{
					Vector4 colour = Utils::GenColour(y * 0.2f, 0.7f);
					Object* cube = Utils::BuildCuboidObject("", Vector3(x - y * 0.5f + 5.0f, 3.5f - y, 4.0f), Vector3(0.5f, 0.5f, 0.5f), 1.f, true, true, colour);
					cube->Physics()->SetFriction(1.0f);
					this->AddGameObject(cube);
				}
			}
		}
		
		//ELASTICITY
		{
			//Sphere Bounce-Pad
			Object* obj = Utils::BuildCuboidObject("BouncePad", Vector3(-2.5f, 0.0f, 6.0f), Vector3(5.0f, 1.0f, 2.0f), 0.0f, true, false, Vector4(0.2f, 0.5f, 1.0f, 1.0f));
			obj->Physics()->SetFriction(1.0f);
			obj->Physics()->SetElasticity(1.0f);
			this->AddGameObject(obj);

			//Create Bouncing Spheres
			for (int i = 0; i < 5; ++i)
			{
				Vector4 colour = Utils::GenColour(0.7f + i * 0.05f, 1.0f);
				Object* obj = Utils::BuildSphereObject("", Vector3(-5.0f + i * 1.25f, 5.5f, 6.0f), 0.5f, 1.0f, true, true, colour);
				obj->Physics()->SetFriction(0.1f);
				obj->Physics()->SetElasticity(i * 0.1f + 0.5f);
				this->AddGameObject(obj);
			}
		}

		//FRICTION
		{
			//Create Ramp
			Object* ramp = Utils::BuildCuboidObject("Ramp", Vector3(4.0f, 3.5f, -5.0f), Vector3(5.0f, 0.5f, 4.0f), 0.0f, true, false, Vector4(1.0f, 0.7f, 1.0f, 1.0f));
			ramp->Physics()->SetOrientation(Quaternion::AxisAngleToQuaterion(Vector3(0.0f, 0.0f, 1.0f), 20.0f));
			ramp->Physics()->SetFriction(1.0f);
			this->AddGameObject(ramp);

			//Create Cubes to roll on ramp
			for (int i = 0; i < 5; ++i)
			{
				Vector4 colour = Vector4(i * 0.25f, 0.7f, (2 - i) * 0.25f, 1.0f);
				Object* cube = Utils::BuildCuboidObject("", Vector3(8.0f, 6.0f, -7.0f + i * 1.1f), Vector3(0.5f, 0.5f, 0.5f), 1.f, true, true, colour);
				cube->Physics()->SetFriction(i * 0.05f);
				cube->Physics()->SetOrientation(Quaternion::AxisAngleToQuaterion(Vector3(0.0f, 0.0f, 1.0f), 200.0f));
				this->AddGameObject(cube);
			}
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