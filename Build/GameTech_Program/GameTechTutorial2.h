
#pragma once

#include <ncltech\Scene.h>
#include <ncltech\SceneManager.h>
#include <ncltech\PhysicsEngine.h>
#include <ncltech\NCLDebug.h>
#include <ncltech\DistanceConstraint.h>
#include "Utils.h"

class GameTechTutorial2 : public Scene
{
public:
	GameTechTutorial2(const std::string& friendly_name)
		: Scene(friendly_name)
	{
	}

	virtual void OnInitializeScene() override
	{
		SceneManager::Instance()->GetCamera()->SetPosition(Vector3(-3.0f, 10.0f, 10.0f));
		SceneManager::Instance()->GetCamera()->SetPitch(-20.f);
		PhysicsEngine::Instance()->SetDebugDrawFlags(DEBUHDRAW_FLAGS_CONSTRAINT);

		//Create Ground
		Object* ground = Utils::BuildCuboidObject("Ground", Vector3(0.0f, 0.0f, 0.0f), Vector3(20.0f, 1.0f, 20.0f), 0.0f, false, false, Vector4(0.2f, 0.5f, 1.0f, 1.0f));
		this->AddGameObject(ground);


		//Create Rope (Single Constraint)
		Object* prevRopeNode = NULL;
		for (int i = 0; i < 5; i++)
		{
			Object* ropeNode = Utils::BuildSphereObject("",
				Vector3(i * 1.1f, 7.f, 0.0f),
				0.5f,
				(prevRopeNode == NULL) ? 0.0f : 1.0f,
				true,
				true,
				Vector4(1.0f, 0.2f, 0.5f, 1.0f));

			//Delete Collision Shape (Incase further tutorials have already been implemented)
			delete ropeNode->Physics()->GetCollisionShape();
			ropeNode->Physics()->SetCollisionShape(NULL);

			if (prevRopeNode != NULL)
			{
				DistanceConstraint* constraint = new DistanceConstraint(prevRopeNode->Physics(), ropeNode->Physics(), Vector3((i - 1) * 1.1f, 7.f, 0.0f), Vector3(i * 1.1f, 7.f, 0.0f));
				PhysicsEngine::Instance()->AddConstraint(constraint);
			}

			this->AddGameObject(ropeNode);
			prevRopeNode = ropeNode;
		}


		//Create Bridge (Double Constraint + Rotation)
			Object* prevBridgeNode = NULL;
			Vector3 bridgeStart = Vector3(-10.f, 5.f, -5.f);
			for (int i = 0; i < 10; i++)
			{
				Object* bridgeNode = Utils::BuildCuboidObject("",
					Vector3(i * 1.1f, 0.0f, 0.0f) + bridgeStart,
					Vector3(0.5f, 0.1f, 1.0f),
					(i == 0 || i == 9) ? 0.0f : 1.0f,
					true,
					true,
					Vector4(1.0f, 0.2f, 0.5f, 1.0f));

				//Delete Collision Shape (Incase further tutorials have already been implemented)
				delete bridgeNode->Physics()->GetCollisionShape();
				bridgeNode->Physics()->SetCollisionShape(NULL);

				if (prevBridgeNode != NULL)
				{
					PhysicsEngine::Instance()->AddConstraint(new DistanceConstraint(
						prevBridgeNode->Physics(),
						bridgeNode->Physics(),
						Vector3((i - 1) * 1.1f + 0.5f, 0.0f, 1.0f) + bridgeStart,
						Vector3(i * 1.1f - 0.5f, 0.0f, 1.0f) + bridgeStart));

					PhysicsEngine::Instance()->AddConstraint(new DistanceConstraint(
						prevBridgeNode->Physics(),
						bridgeNode->Physics(),
						Vector3((i - 1) * 1.1f + 0.5f, 0.0f, -1.0f) + bridgeStart,
						Vector3(i * 1.1f - 0.5f, 0.0f, -1.0f) + bridgeStart));
				}

				this->AddGameObject(bridgeNode);
				prevBridgeNode = bridgeNode;
			}

		
	}

	virtual void OnUpdateScene(float dt) override
	{
		Scene::OnUpdateScene(dt);

		bool drawConstraints = PhysicsEngine::Instance()->GetDebugDrawFlags() & DEBUHDRAW_FLAGS_CONSTRAINT;

		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "Physics:");
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "     Draw Constraints : %s (Press C to toggle)", drawConstraints ? "Enabled" : "Disabled");

		if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_C))
		{
			drawConstraints = !drawConstraints;
		}

		PhysicsEngine::Instance()->SetDebugDrawFlags(drawConstraints ? DEBUHDRAW_FLAGS_CONSTRAINT : NULL);

	}
};