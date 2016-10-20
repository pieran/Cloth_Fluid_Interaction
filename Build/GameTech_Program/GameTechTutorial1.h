
#pragma once

#include <ncltech\Scene.h>
#include <ncltech\SceneRenderer.h>
#include <ncltech\PhysicsEngine.h>
#include <ncltech\NCLDebug.h>
#include <ncltech\ObjectMesh.h>
#include <ncltech\CommonMeshes.h>
#include "Utils.h"

class GameTechTutorial1 : public Scene
{
public:
	GameTechTutorial1(const std::string& friendly_name)
		: Scene(friendly_name)
	{
		m_TargetTexture = SOIL_load_OGL_texture(TEXTUREDIR"target.tga",
			SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID,
			SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT);

		glBindTexture(GL_TEXTURE_2D, m_TargetTexture);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);

		SetWorldRadius(10.0f);
	}

	~GameTechTutorial1()
	{
		if (m_TargetTexture)
		{
			glDeleteTextures(1, &m_TargetTexture);
			m_TargetTexture = NULL;
		}
	}

	virtual void OnInitializeScene() override
	{
		m_TrajectoryPoints.clear();

		PhysicsEngine::Instance()->SetPaused(false);

		SceneManager::Instance()->GetCamera()->SetPosition(Vector3(-6.25f, 2.0f, 10.0f));
		SceneManager::Instance()->GetCamera()->SetPitch(0.0f);
		SceneManager::Instance()->GetCamera()->SetYaw(0.0f);

		PhysicsEngine::Instance()->SetGravity(Vector3(0.0f, 0.0f, 0.0f));		//No Gravity
		PhysicsEngine::Instance()->SetDampingFactor(1.0f);						//No Damping



		//Create Ground
		this->AddGameObject(Utils::BuildCuboidObject("Ground", Vector3(-6.25f, -0.2f, 0.0f), Vector3(10.0f, 0.1f, 2.f), 0.0f, false, false, Vector4(0.2f, 1.0f, 0.5f, 1.0f)));


		//Create Target
		ObjectMesh* target = new ObjectMesh("Target");
		target->SetMesh(CommonMeshes::Cube(), false);
		target->SetTexture(m_TargetTexture, false);
		target->SetLocalTransform(Matrix4::Translation(Vector3(0.1f, 2.0f, 0.0f)) * Matrix4::Scale(Vector3(0.1f, 2.0f, 2.f)));
		target->SetColour(Vector4(1.0f, 1.0f, 1.0f, 1.0f));
		target->SetBoundingRadius(4.0f);
		this->AddGameObject(target);


		//Create a projectile
		m_Sphere = new ObjectMesh("Sphere");
		m_Sphere->SetMesh(CommonMeshes::Sphere(), false);
		m_Sphere->SetLocalTransform(Matrix4::Scale(Vector3(0.5f, 0.5f, 0.5f)));
		m_Sphere->SetColour(Vector4(1.0f, 0.2f, 0.5f, 1.0f));
		m_Sphere->SetBoundingRadius(1.0f);

		m_Sphere->CreatePhysicsNode();
		m_Sphere->Physics()->SetInverseMass(1.f);
		this->AddGameObject(m_Sphere);

		ResetScene(PhysicsEngine::Instance()->GetUpdateTimestep());
	}

	void ResetScene(float timestep)
	{
		PhysicsEngine::Instance()->SetUpdateTimestep(timestep);
		PhysicsEngine::Instance()->SetPaused(false);
		m_TrajectoryPoints.clear();
		m_Sphere->Physics()->SetPosition(Vector3(-12.5f, 2.0f, 0.f));
		m_Sphere->Physics()->SetLinearVelocity(Vector3(0.f, 2.5f, 0.0f));
		m_Sphere->Physics()->SetForce(Vector3(1.f, -1.f, 0.0f));
	}

	virtual void GameTechTutorial1::OnUpdateScene(float dt) override
	{
		Scene::OnUpdateScene(dt);

		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "Physics Timestep: %5.2fms [%5.2ffps]",
			PhysicsEngine::Instance()->GetUpdateTimestep() * 1000.0f,
			1.0f / PhysicsEngine::Instance()->GetUpdateTimestep()
			);
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "Select Integration Timestep:");
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "     1: 5fps");
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "     2: 15fps");
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "     3: 30fps");
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "     4: 60fps");

		if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_1))	ResetScene(1.0f / 5.0f);
		if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_2))	ResetScene(1.0f / 15.0f);
		if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_3))	ResetScene(1.0f / 30.0f);
		if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_4))	ResetScene(1.0f / 60.0f);

		if (!PhysicsEngine::Instance()->IsPaused())
		{
			m_TrajectoryPoints.push_back(m_Sphere->Physics()->GetPosition());
		}

		if (m_Sphere->Physics()->GetPosition().y < 0.0f)
		{
			PhysicsEngine::Instance()->SetPaused(true);
		}

		for (size_t i = 1; i < m_TrajectoryPoints.size(); i++)
		{
			NCLDebug::DrawThickLine(m_TrajectoryPoints[i - 1], m_TrajectoryPoints[i], 0.05f, Vector4(1.0f, 0.0f, 0.0f, 1.0f));
		}
	}

private:
	GLuint					m_TargetTexture;
	ObjectMesh*				m_Sphere;
	std::vector<Vector3>	m_TrajectoryPoints;
};