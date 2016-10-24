
#include "ClothScene1.h"

#include <nclgl\Vector4.h>
#include <ncltech\PhysicsEngine.h>
#include <ncltech\DistanceConstraint.h>
#include <ncltech\SceneManager.h>
#include "CollidableObject.h"
#include <ncltech\CommonMeshes.h>

#include <ncltech\Utils.h>
using namespace Utils;

ClothScene1::ClothScene1(const std::string& friendly_name)
	: Scene(friendly_name)
	, m_Sim(NULL)
	, m_ShadowCycleKey(5)
	, m_SuperSampleCycleKey(1)
	, m_ClothTexFront(0)
	, m_ClothTexBack(0)
{
	auto load_font = [&](const char* filename)
	{
		GLuint tex = SOIL_load_OGL_texture(filename,
			SOIL_LOAD_AUTO, SOIL_CREATE_NEW_ID,
			SOIL_FLAG_MIPMAPS | SOIL_FLAG_INVERT_Y | SOIL_FLAG_NTSC_SAFE_RGB | SOIL_FLAG_COMPRESS_TO_DXT);

		glBindTexture(GL_TEXTURE_2D, tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
		return tex;
	};

	m_ClothTexBack = load_font(TEXTUREDIR"cloth_seamless_blue.png");
	m_ClothTexFront = load_font(TEXTUREDIR"cloth_seamless_yellow.png");

}

ClothScene1::~ClothScene1()
{
	if (m_Sim)
	{
		delete m_Sim;
		m_Sim = NULL;
	}

	if (m_ClothTexFront)
	{
		glDeleteTextures(1, &m_ClothTexFront);
		m_ClothTexFront = NULL;
	}

	if (m_ClothTexBack)
	{
		glDeleteTextures(1, &m_ClothTexBack);
		m_ClothTexBack = NULL;
	}
}


void ClothScene1::OnInitializeScene()
{
	PhysicsEngine::Instance()->SetPaused(true);

	m_ShadowCycleKey = 5;
	m_SuperSampleCycleKey = 1;

	SceneManager::Instance()->GetCamera()->SetPosition(Vector3(2.5f, 7.0f, -2.5f));
	SceneManager::Instance()->GetCamera()->SetYaw(145.f);
	SceneManager::Instance()->GetCamera()->SetPitch(-20.f);
	SceneManager::Instance()->SetSpecularIntensity(16.0f);
	
	SceneManager::Instance()->SetShadowMapNum(8);
	SceneManager::Instance()->SetShadowMapSize(2048);

	this->SetWorldRadius(5.f);

	//Create Ground
	this->AddGameObject(BuildCuboidObject("Ground", Vector3(0.0f, -1.0f, 0.0f), Vector3(20.0f, 1.0f, 20.0f), 0.0f, true, false, Vector4(0.2f, 0.5f, 1.0f, 1.0f)));

	Matrix4 transform = Matrix4::Translation(Vector3(0.0f, 1.2f, 0.0f)) * Matrix4::Rotation(90.0f, Vector3(1.0f, 0.0f, 0.0f));
	m_Sim = new XPBD();
	m_Sim->InitializeCloth(129, 129, transform);
	m_Sim->SetTextureFront(m_ClothTexFront, false);
	m_Sim->SetTextureBack(m_ClothTexBack, false);
	m_Sim->SetLocalTransform(Matrix4());
	m_Sim->SetColour(Vector4(1.0f, 1.0f, 1.0f, 1.0f));
	m_Sim->SetBoundingRadius(10000.0f);
	this->AddGameObject(m_Sim);


	auto add_sphere = [&](const Vector3& pos, float radius)
	{
		//Object* sphere = BuildSphereObject("ColSphere", pos, radius - 0.003f, 0.0f, false, true, );

		XPBDSphereConstraint sc;
		memcpy(&sc.centre, &pos, sizeof(float3));
		sc.radius = radius;
		m_Sim->m_SphereConstraints.push_back(sc);

		CollidableObject* sphere = new CollidableObject("ColSphere", &m_Sim->m_SphereConstraints[m_Sim->m_SphereConstraints.size()-1]);

		sphere->SetMesh(CommonMeshes::Sphere(), false);
		sphere->SetTexture(CommonMeshes::CheckerboardTex(), false);
		sphere->SetLocalTransform(Matrix4::Scale(Vector3(radius * 0.99, radius * 0.99, radius * 0.99)));
		sphere->SetColour(Vector4(1.0f, 0.8f, 0.8f, 1.0f));
		sphere->SetBoundingRadius(radius * 0.99);
		sphere->SetLocalTransform(Matrix4::Translation(pos) * sphere->GetLocalTransform());

		m_Spheres.push_back(sphere);
		this->AddGameObject(sphere);
	};

	add_sphere(Vector3(0.0f, 0.6f, 0.15f), 0.15f);

	m_Sim->UpdateGLBuffers();
}

void ClothScene1::OnCleanupScene()
{
	Scene::OnCleanupScene(); //Just delete all created game objects
}

void ClothScene1::OnUpdateScene(float dt)
{
	/*for (size_t i = 0; i < m_Spheres.size(); ++i)
	{
		memcpy(&m_Sim->m_SphereConstraints[i].centre, &m_Spheres[i]->GetWorldTransform().GetPositionVector(), sizeof(float3));
	}*/

	Scene::OnUpdateScene(dt);

	if (Window::GetKeyboard()->KeyDown(KEYBOARD_G))
	{
		m_Sim->Update(1.0f / 60.0f);
		m_Sim->UpdateGLBuffers();
	}

	float gamma = SceneManager::Instance()->GetGammaCorrection();

	const Vector4 status_colour = Vector4(1.0f, 0.9f, 0.8f, 1.0f);
	NCLDebug::AddStatusEntry(status_colour, "Shadow Cascades: %d @ %dx%d (Press G/H to cycle)",
		SceneManager::Instance()->GetShadowMapNum(),
		SceneManager::Instance()->GetShadowMapSize(),
		SceneManager::Instance()->GetShadowMapSize());

	NCLDebug::AddStatusEntry(status_colour, "Super Sampling: %3.0f%% (Press I/O to cycle)", SceneManager::Instance()->GetSuperSamplingScalar() * 100.0f);
	NCLDebug::AddStatusEntry(status_colour, "Gamma Correction : %5.2f (5/6 to change)", gamma);



	uint newShadowKey = m_ShadowCycleKey;
	uint newSuperSampleKey = m_SuperSampleCycleKey;

	/*if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_G))
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
		gamma += dt * 0.2f;*/

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


