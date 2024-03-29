
#include "GraphicsScene.h"

#include <nclgl\Vector4.h>
#include <ncltech\PhysicsEngine.h>
#include <ncltech\DistanceConstraint.h>
#include <ncltech\SceneManager.h>


#include "Utils.h"
using namespace Utils;

GraphicsScene::GraphicsScene(const std::string& friendly_name)
	: Scene(friendly_name)
	, m_AccumTime(0.0f)
	, m_ShadowCycleKey(5)
	, m_SuperSampleCycleKey(1)
{
}

GraphicsScene::~GraphicsScene()
{

}


void GraphicsScene::OnInitializeScene()
{
	PhysicsEngine::Instance()->SetPaused(true);

	m_ShadowCycleKey = 5;
	m_SuperSampleCycleKey = 1;

	SceneManager::Instance()->GetCamera()->SetPosition(Vector3(15.0f, 10.0f, -15.0f));
	SceneManager::Instance()->GetCamera()->SetYaw(140.f);
	SceneManager::Instance()->GetCamera()->SetPitch(-20.f);


	//Create Ground
	this->AddGameObject(BuildCuboidObject("Ground", Vector3(0.0f, -1.0f, 0.0f), Vector3(20.0f, 1.0f, 20.0f), 0.0f, true, false, Vector4(0.2f, 0.5f, 1.0f, 1.0f)));

	auto create_cube_tower = [&](const Vector3& offset, float cubewidth)
	{
		for (int x = 0; x < 2; ++x)
		{
			for (int y = 0; y < 5; ++y)
			{
				uint idx = x * 5 + y;
				Vector4 colour = GenColour(idx / 10.f, 0.5f);
				Object* cube = BuildCuboidObject("", offset + Vector3(x * cubewidth, y * cubewidth, cubewidth * (idx % 2 == 0) ? 0.5f : -0.5f), Vector3(cubewidth, cubewidth, cubewidth) * 0.5f, 0.f, false, true, colour);
				this->AddGameObject(cube);
			}
		}
	};

	auto create_ball_cube = [&](const Vector3& offset, const Vector3& scale, float ballsize)
	{
		for (int x = 0; x < 5; ++x)
		{
			for (int y = 0; y < 5; ++y)
			{
				for (int z = 0; z < 5; ++z)
				{
					Object* sphere = BuildSphereObject("", offset + Vector3(scale.x *x, scale.y * y, scale.z * z), ballsize, 0.f, false, false, Vector4(1.0f, 0.5f, 0.2f, 1.0f));
					this->AddGameObject(sphere);
				}
			}
		}
	};

	//Create Cube Towers
	create_cube_tower(Vector3(3.0f, 0.5f, 3.0f), 1.0f);
	create_cube_tower(Vector3(-3.0f, 0.5f, -3.0f), 1.0f);
	
	//Create Test Ball Pit
	create_ball_cube(Vector3(-8.0f, 0.5f, 12.0f), Vector3(0.5f, 0.5f, 0.5f), 0.1f);
	create_ball_cube(Vector3(8.0f, 0.5f, 12.0f), Vector3(0.3f, 0.3f, 0.3f), 0.1f);
	create_ball_cube(Vector3(-8.0f, 0.5f, -12.0f), Vector3(0.2f, 0.2f, 0.2f), 0.1f);
	create_ball_cube(Vector3(8.0f, 0.5f, -12.0f), Vector3(0.5f, 0.5f, 0.5f), 0.1f);
}

void GraphicsScene::OnCleanupScene()
{
	Scene::OnCleanupScene(); //Just delete all created game objects
}

void GraphicsScene::OnUpdateScene(float dt)
{
	m_AccumTime += dt;

	Vector3 invLightDir = Matrix4::Rotation(15.f * dt, Vector3(0.0f, 1.0f, 0.0f)) * SceneManager::Instance()->GetInverseLightDirection();
	float specular = (sin(m_AccumTime*0.5f) * 0.5f + 0.5f) * 256.f + 4.0f;
	
	SceneManager::Instance()->SetInverseLightDirection(invLightDir);
	SceneManager::Instance()->SetSpecularIntensity(specular);

	float gamma = SceneManager::Instance()->GetGammaCorrection();

	const Vector4 status_colour = Vector4(1.0f, 0.9f, 0.8f, 1.0f);
	NCLDebug::AddStatusEntry(status_colour, "Shadow Cascades: %d @ %dx%d (Press G/H to cycle)",
		SceneManager::Instance()->GetShadowMapNum(),
		SceneManager::Instance()->GetShadowMapSize(),
		SceneManager::Instance()->GetShadowMapSize());

	NCLDebug::AddStatusEntry(status_colour, "Super Sampling: %3.0f%% (Press I/O to cycle)", SceneManager::Instance()->GetSuperSamplingScalar() * 100.0f);
	NCLDebug::AddStatusEntry(status_colour, "Gamma Correction : %5.2f (5/6 to change)", gamma);
	NCLDebug::AddStatusEntry(status_colour, "Scene Light Direction   : [%4.2f, %5.2f, %5.2f]", -invLightDir.x, -invLightDir.y, -invLightDir.z);
	NCLDebug::AddStatusEntry(status_colour, "Scene Specular Exponent : %5.2f", specular);
	


	uint newShadowKey = m_ShadowCycleKey;
	uint newSuperSampleKey = m_SuperSampleCycleKey;

	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_G))
		newShadowKey = ((newShadowKey + 1) % 9);

	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_H))
		newShadowKey = ((newShadowKey == 0 ? 9 : newShadowKey) - 1);


	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_I))
		newSuperSampleKey = ((newSuperSampleKey + 1) % 5);

	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_O))
		newSuperSampleKey = ((newSuperSampleKey == 0 ? 5 : m_SuperSampleCycleKey) - 1);

	
	if (Window::GetKeyboard()->KeyDown(KEYBOARD_5))
		gamma -= dt * 0.2f;
	if (Window::GetKeyboard()->KeyDown(KEYBOARD_6))
		gamma += dt * 0.2f;

	gamma = min(max(gamma, 0.0f), 5.0f);
	SceneManager::Instance()->SetGammaCorrection(gamma);


	if (newSuperSampleKey != m_SuperSampleCycleKey)
	{
		m_SuperSampleCycleKey = newSuperSampleKey;
		switch (m_SuperSampleCycleKey)
		{
		case 0:
			SceneManager::Instance()->SetSuperSamplingScalar(1.0f);
			break;
		case 1:
			SceneManager::Instance()->SetSuperSamplingScalar(1.5f);
			break;
		case 2:
			SceneManager::Instance()->SetSuperSamplingScalar(2.0f);
			break;
		case 3:
			SceneManager::Instance()->SetSuperSamplingScalar(4.0f);
			break;
		default:
			SceneManager::Instance()->SetSuperSamplingScalar(8.0f);
			break;
		}
	}

	if (newShadowKey != m_ShadowCycleKey)
	{
		m_ShadowCycleKey = newShadowKey;
		switch (m_ShadowCycleKey)
		{
		case 0:
			SceneManager::Instance()->SetShadowMapNum(0);
			SceneManager::Instance()->SetShadowMapSize(1024);
			break;
		case 1:
			SceneManager::Instance()->SetShadowMapNum(1);
			SceneManager::Instance()->SetShadowMapSize(2048);
			break;
		case 2:
			SceneManager::Instance()->SetShadowMapNum(2);
			SceneManager::Instance()->SetShadowMapSize(1024);
			break;
		case 3:
			SceneManager::Instance()->SetShadowMapNum(2);
			SceneManager::Instance()->SetShadowMapSize(2048);
			break;
		case 4:
			SceneManager::Instance()->SetShadowMapNum(4);
			SceneManager::Instance()->SetShadowMapSize(1024);
			break;
		case 5:
			SceneManager::Instance()->SetShadowMapNum(4);
			SceneManager::Instance()->SetShadowMapSize(2048);
			break;
		case 6:
			SceneManager::Instance()->SetShadowMapNum(8);
			SceneManager::Instance()->SetShadowMapSize(2048);
			break;
		case 7:
			SceneManager::Instance()->SetShadowMapNum(12);
			SceneManager::Instance()->SetShadowMapSize(2048);
			break;
		default:
			SceneManager::Instance()->SetShadowMapNum(16);
			SceneManager::Instance()->SetShadowMapSize(2048);
			break;
		}
	}
}


