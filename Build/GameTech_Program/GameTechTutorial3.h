
#pragma once

#include <nclgl\OBJMesh.h>
#include <ncltech\Scene.h>
#include <ncltech\SceneManager.h>
#include <ncltech\PhysicsEngine.h>
#include <ncltech\NCLDebug.h>
#include <ncltech\ObjectMesh.h>
#include <ncltech\SphereCollisionShape.h>
#include <ncltech\CuboidCollisionShape.h>
#include "ObjectPlayer.h"
#include "Utils.h"

class GameTechTutorial3 : public Scene
{
public:
	GameTechTutorial3(const std::string& friendly_name)
		: Scene(friendly_name)
		, m_MeshHouse(NULL)
		, m_MeshGarden(NULL)
	{
		glGenTextures(1, &m_whiteTexture);
		glBindTexture(GL_TEXTURE_2D, m_whiteTexture);
		int white_pixel = 0xFFFFFFFF;
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, &white_pixel);

		m_MeshHouse = new OBJMesh(MESHDIR"house.obj");
		m_MeshGarden = new OBJMesh(MESHDIR"garden.obj");

		m_MeshPlayer = new OBJMesh(MESHDIR"raptor.obj");
		m_MeshPlayer->GenerateNormals();
		GLuint dTex, nTex;

		dTex = SOIL_load_OGL_texture(TEXTUREDIR"raptor.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT);
		glBindTexture(GL_TEXTURE_2D, dTex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST); //No linear interpolation to get crisp checkerboard no matter the scalling
		glBindTexture(GL_TEXTURE_2D, 0);

		nTex = SOIL_load_OGL_texture(TEXTUREDIR"raptor_normal.jpg", SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID, SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT);
		glBindTexture(GL_TEXTURE_2D, nTex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST); //No linear interpolation to get crisp checkerboard no matter the scalling
		glBindTexture(GL_TEXTURE_2D, 0);

		m_MeshPlayer->SetTexture(dTex);
		m_MeshPlayer->SetBumpMap(nTex);
	}

	virtual ~GameTechTutorial3()
	{
		if (m_whiteTexture)
		{
			glDeleteTextures(1, &m_whiteTexture);
			m_whiteTexture = NULL;
		}

		if (m_MeshHouse)
		{
			m_MeshHouse->SetTexture(NULL);
			delete m_MeshHouse;
			m_MeshHouse = NULL;
		}

		if (m_MeshGarden)
		{
			m_MeshGarden->SetTexture(NULL);
			delete m_MeshGarden;
			m_MeshGarden = NULL;
		}

		if (m_MeshPlayer)
		{
			delete m_MeshPlayer;
			m_MeshPlayer = NULL;
		}

	}

	virtual void OnInitializeScene() override
	{
		PhysicsEngine::Instance()->SetDebugDrawFlags(DEBUHDRAW_FLAGS_COLLISIONNORMALS | DEBUHDRAW_FLAGS_COLLISIONVOLUMES);

		SceneManager::Instance()->GetCamera()->SetPosition(Vector3(-3.0f, 10.0f, 15.0f));
		SceneManager::Instance()->GetCamera()->SetYaw(-10.f);
		SceneManager::Instance()->GetCamera()->SetPitch(-30.f);

		//Create Ground
		this->AddGameObject(Utils::BuildCuboidObject("Ground", Vector3(0.0f, -1.001f, 0.0f), Vector3(20.0f, 1.0f, 20.0f), 0.0f, false, false, Vector4(0.2f, 0.5f, 1.0f, 1.0f)));

		//Create Player
		ObjectPlayer* player = new ObjectPlayer("Player1");
		player->SetMesh(m_MeshPlayer, false);
		player->CreatePhysicsNode();
		player->Physics()->SetPosition(Vector3(0.0f, 0.5f, 0.0f));
		player->Physics()->SetCollisionShape(new CuboidCollisionShape(Vector3(0.5f, 0.5f, 1.0f)));
		player->SetBoundingRadius(1.0f);
		player->SetColour(Vector4(1.0f, 1.0f, 1.0f, 1.0f));
		this->AddGameObject(player);


		//Create Some Objects
		{
			const Vector3 col_size = Vector3(2.0f, 2.f, 2.f);
			ObjectMesh* obj = new ObjectMesh("House");
			obj->SetLocalTransform(Matrix4::Scale(Vector3(2.0f, 2.0f, 2.f)));
			obj->SetMesh(m_MeshHouse, false);
			obj->SetTexture(m_whiteTexture, false);
			obj->SetColour(Vector4(0.8f, 0.3f, 0.1f, 1.0f));
			obj->SetBoundingRadius(col_size.Length());
			obj->CreatePhysicsNode();
			obj->Physics()->SetPosition(Vector3(-5.0f, 2.f, -5.0f));
			obj->Physics()->SetCollisionShape(new CuboidCollisionShape(col_size));
			obj->Physics()->SetOnCollisionCallback([](PhysicsObject* collidingObject){
				NCLDebug::Log(Vector3(0.6f, 0.3f, 0.1f), "You are inside the house!");
				return false;
			});

			this->AddGameObject(obj);
		}

		{
			const Vector3 col_size = Vector3(2.0f, 0.5f, 2.f);
			ObjectMesh* obj = new ObjectMesh("Garden");
			obj->SetLocalTransform(Matrix4::Scale(Vector3(2.0f, 1.0f, 2.f)));
			obj->SetMesh(m_MeshGarden, false);
			obj->SetTexture(m_whiteTexture, false);
			obj->SetColour(Vector4(0.5f, 1.0f, 0.5f, 1.0f));
			obj->SetBoundingRadius(col_size.Length());
			obj->CreatePhysicsNode();
			obj->Physics()->SetPosition(Vector3(5.0f, 0.5f, -5.0f));
			obj->Physics()->SetCollisionShape(new CuboidCollisionShape(col_size));
			obj->Physics()->SetOnCollisionCallback([](PhysicsObject* collidingObject){
				NCLDebug::Log(Vector3(0.0f, 1.0f, 0.0f), "You are inside the garden!");
				return false;
			});

			this->AddGameObject(obj);
		}

		//'Hidden' Physics Node (without Scene-GameObject)
		{
			PhysicsObject* obj = new PhysicsObject();
			obj->SetPosition(Vector3(5.0f, 1.0f, 0.0f));
			obj->SetCollisionShape(new SphereCollisionShape(1.0f));
			obj->SetOnCollisionCallback(
				[obj](PhysicsObject* collidingObject) {

				NCLDebug::Log(Vector3(1.0f, 0.0f, 0.0f), "You found the secret!");

				float r_x = 5.f * ((rand() % 200) / 100.f - 1.0f);
				float r_z = 3.f * ((rand() % 200) / 100.f - 1.0f);
				obj->SetPosition(Vector3(r_x, 1.0f, r_z + 3.0f));
				return false;
			});
			PhysicsEngine::Instance()->AddPhysicsObject(obj);
		}
	}


	virtual void OnUpdateScene(float dt) override
	{
		Scene::OnUpdateScene(dt);

		uint drawFlags = PhysicsEngine::Instance()->GetDebugDrawFlags();

		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "Physics:");
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "     (Arrow Keys to Move Player)");
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "     Draw Collision Volumes : %s (Press C to toggle)", (drawFlags & DEBUHDRAW_FLAGS_COLLISIONVOLUMES) ? "Enabled" : "Disabled");
		NCLDebug::AddStatusEntry(Vector4(1.0f, 0.9f, 0.8f, 1.0f), "     Draw Collision Normals : %s (Press N to toggle)", (drawFlags & DEBUHDRAW_FLAGS_COLLISIONNORMALS) ? "Enabled" : "Disabled");

		if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_C))
			drawFlags ^= DEBUHDRAW_FLAGS_COLLISIONVOLUMES;
		
		if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_N))
			drawFlags ^= DEBUHDRAW_FLAGS_COLLISIONNORMALS;

		PhysicsEngine::Instance()->SetDebugDrawFlags(drawFlags);

	}

private:
	OBJMesh *m_MeshHouse, *m_MeshGarden;
	GLuint	m_whiteTexture;
	OBJMesh* m_MeshPlayer;
};