
#include "FluidScene1.h"

#include <nclgl\Vector4.h>
#include <ncltech\PhysicsEngine.h>
#include <ncltech\DistanceConstraint.h>
#include <ncltech\SceneManager.h>
#include <ncltech\CommonMeshes.h>

#include <random>

#include <ncltech\Utils.h>
using namespace Utils;

FluidScene1::FluidScene1(const std::string& friendly_name)
	: Scene(friendly_name)
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


}

FluidScene1::~FluidScene1()
{
	if (fluid)
	{
		delete fluid;
		fluid = NULL;
	}
}


void FluidScene1::OnInitializeScene()
{



	PhysicsEngine::Instance()->SetPaused(true);

	SceneManager::Instance()->GetCamera()->SetPosition(Vector3(2.5f, 7.0f, -2.5f));
	SceneManager::Instance()->GetCamera()->SetYaw(145.f);
	SceneManager::Instance()->GetCamera()->SetPitch(-20.f);
	SceneManager::Instance()->SetSpecularIntensity(16.0f);

	SceneManager::Instance()->SetShadowMapNum(8);
	SceneManager::Instance()->SetShadowMapSize(2048);

	this->SetWorldRadius(5.f);

	//Create Ground
	this->AddGameObject(BuildCuboidObject("Ground", Vector3(0.0f, -1.0f, 0.0f), Vector3(20.0f, 1.0f, 20.0f), 0.0f, true, false, Vector4(0.2f, 0.5f, 1.0f, 1.0f)));




	//Create Fluid
	std::vector<float> particles;
	std::vector<float> boundary_particles;

	//Set Boundary Size
	Vector3 bmin = Vector3(-3.0f, 0.0f, -1.0f);
	Vector3 bmax = Vector3(2.0f, 2.0f, 3.0f);
	AddBoundaryMarkers(boundary_particles, bmin, bmax);


	//Create Fluid
	GenerateFluidCube_Random(
		particles,
		Vector3(-3.0f, 0.0f, -1.0f),
		Vector3(-1.0f, 2.0f, 1.0f),
		196656);

	fluid = new FluidPicFlip();
	fluid->allocate_buffers(particles, boundary_particles);





	fluid_renderer = new FluidSSRenderer();
	fluid_renderer->Initialize();

	fluid_renderer->SetWorldSpace(Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 1.0f, 1.0f));
	fluid_renderer->m_ParticleRadius = 0.0225f;
	fluid_renderer->m_NumParticles = particles.size() / 3;
	fluid_renderer->BuildBuffers(particles.size() / 3);
	fluid_renderer->SetBackgroundSceneCubemap(NULL);

	fluid_renderer->m_NumParticles = particles.size() / 3;

	fluid->set_ogl_vbo(fluid_renderer->m_glVerts, fluid_renderer->m_glVel, fluid_renderer->m_glDensity, fluid_renderer->m_glPressure);
	fluid->copy_arrays_to_ogl_vbos();
	SceneManager::Instance()->AddPostProcessEffect(fluid_renderer);
}

void FluidScene1::OnCleanupScene()
{
	Scene::OnCleanupScene(); //Just delete all created game objects
}

void FluidScene1::OnUpdateScene(float dt)
{
	Scene::OnUpdateScene(dt);

	if (Window::GetKeyboard()->KeyDown(KEYBOARD_G))
	{
		fluid->update();
		fluid->copy_arrays_to_ogl_vbos();
	}
	
}


void FluidScene1::GenerateFluidCube_Random(std::vector<float>& particles, const Vector3& min, const Vector3& max, size_t num)
{
	std::uniform_real_distribution<float> d_x(min.x, max.x);
	std::uniform_real_distribution<float> d_y(min.y, max.y);
	std::uniform_real_distribution<float> d_z(min.z, max.z);

	for (size_t i = 0; i < num; i++) {
		particles.push_back(d_x(generator));
		particles.push_back(d_y(generator));
		particles.push_back(d_z(generator));
	}
}


void FluidScene1::AddBoundaryMarkers(std::vector<float>& boundary_particles, const Vector3& bmin, const Vector3& bmax)
{
	boundary_particles.push_back(bmin.x);
	boundary_particles.push_back(bmin.y);
	boundary_particles.push_back(bmin.z);

	boundary_particles.push_back(bmin.x);
	boundary_particles.push_back(bmax.y);
	boundary_particles.push_back(bmin.z);

	boundary_particles.push_back(bmax.x);
	boundary_particles.push_back(bmin.y);
	boundary_particles.push_back(bmin.z);

	boundary_particles.push_back(bmax.x);
	boundary_particles.push_back(bmax.y);
	boundary_particles.push_back(bmin.z);

	boundary_particles.push_back(bmin.x);
	boundary_particles.push_back(bmin.y);
	boundary_particles.push_back(bmax.z);

	boundary_particles.push_back(bmin.x);
	boundary_particles.push_back(bmax.y);
	boundary_particles.push_back(bmax.z);

	boundary_particles.push_back(bmax.x);
	boundary_particles.push_back(bmin.y);
	boundary_particles.push_back(bmax.z);

	boundary_particles.push_back(bmax.x);
	boundary_particles.push_back(bmax.y);
	boundary_particles.push_back(bmax.z);
}
