
#include "FluidClothScene1.h"

#include <nclgl\Vector4.h>
#include <ncltech\PhysicsEngine.h>
#include <ncltech\DistanceConstraint.h>
#include <ncltech\SceneManager.h>
#include <ncltech\CommonMeshes.h>

#include <random>

#include <ncltech\Utils.h>
using namespace Utils;

FluidClothScene1::FluidClothScene1(const std::string& friendly_name)
: Scene(friendly_name)
, coupler(NULL)
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

FluidClothScene1::~FluidClothScene1()
{
	if (fluid)
	{
		delete fluid;
		fluid = NULL;
	}

	if (m_Sim)
	{
		delete m_Sim;
		m_Sim = NULL;
	}

	if (coupler)
	{
		delete coupler;
		coupler = NULL;
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


void FluidClothScene1::OnInitializeScene()
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



	m_Timestep = 1.0f / 360.0f;

	//Create Fluid
	fluid = new FluidPicFlip();
	fluid->m_Params.dt = m_Timestep;


	std::vector<float> particles;
	std::vector<float> boundary_particles;

	//Set Boundary Size
	Vector3 bmin = Vector3(-5.0f, 0.0f, -5.0f);
	Vector3 bmax = Vector3(5.0f, 10.0f, 5.0f);
	Vector3 bbsize = bmax - bmin;
	AddBoundaryMarkers(boundary_particles, bmin, bmax);

	float density = 128.f; //Particles Per Cell
	fluid->m_Params.particles_per_cell = density;

	float grid_y = round(fluid->m_Params.grid_resolution.x * (bbsize.y / bbsize.x));
	float grid_z = round(fluid->m_Params.grid_resolution.x * (bbsize.z / bbsize.x));
	Vector3 wrld2Grid = Vector3(fluid->m_Params.grid_resolution.x, grid_y, grid_z) / bbsize;


	//Create Fluid
	GenerateFluidCube_Random(
		particles,
		Vector3(-1.0f, 3.0f, -1.0f),
		Vector3(1.0f, 4.5f, 1.0f),
		density,
		wrld2Grid);

	
	fluid->allocate_buffers(particles, boundary_particles);
	



	float particle_size = 0.0125f;

	fluid_renderer = new FluidSSRenderer();
	fluid_renderer->Initialize();

	fluid_renderer->SetWorldSpace(Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 1.0f, 1.0f));
	fluid_renderer->m_ParticleRadius = particle_size;
	fluid_renderer->m_NumParticles = particles.size() / 3;
	fluid_renderer->BuildBuffers(particles.size() / 3);
	fluid_renderer->SetBackgroundSceneCubemap(NULL);
	
	fluid_renderer->m_NumParticles = particles.size() / 3;

	fluid->set_ogl_vbo(fluid_renderer->m_glVerts, fluid_renderer->m_glVel, fluid_renderer->m_glDensity, fluid_renderer->m_glPressure);
	fluid->copy_arrays_to_ogl_vbos();
	SceneManager::Instance()->AddPostProcessEffect(fluid_renderer);







	Matrix4 transform = Matrix4::Translation(Vector3(0.0f, 2.6f, 0.0f)) * Matrix4::Rotation(90.0f, Vector3(1.0f, 0.0f, 0.0f)) * Matrix4::Scale(Vector3(4.0f, 4.0f, 4.0f));
	m_Sim = new XPBD();
	m_Sim->InitializeCloth(65, 65, transform);
	m_Sim->SetTextureFront(m_ClothTexFront, false);
	m_Sim->SetTextureBack(m_ClothTexBack, false);
	m_Sim->SetLocalTransform(Matrix4());
	m_Sim->SetColour(Vector4(1.0f, 1.0f, 1.0f, 1.0f));
	m_Sim->SetBoundingRadius(10000.0f);
	this->AddGameObject(m_Sim);

	m_Sim->UpdateGLBuffers();


	coupler = new FluidClothCoupler(m_Sim, fluid);
	coupler->m_Params.particle_size = particle_size * 5.0f;
	coupler->m_Params.particle_iweight = 1.0f / 0.000001f;
}

void FluidClothScene1::OnCleanupScene()
{
	Scene::OnCleanupScene(); //Just delete all created game objects
}

void FluidClothScene1::OnUpdateScene(float dt)
{
	Scene::OnUpdateScene(dt);

	if (Window::GetKeyboard()->KeyDown(KEYBOARD_G))
	{
		for (int i = 0; i < 6; ++i)
		{
			fluid->update();

			m_Sim->Update(m_Timestep);
			m_Sim->GenerateNormals(false);
			coupler->HandleCollisions(m_Timestep);

			fluid->copy_arrays_to_ogl_vbos();
			m_Sim->UpdateGLBuffers();
		}
	}

}


void FluidClothScene1::GenerateFluidCube_Random(std::vector<float>& particles, const Vector3& min, const Vector3& max, float density, const Vector3& grid_dim)
{
	Vector3 span = (max - min) * grid_dim;
	size_t num = span.x * span.y * span.z * density;


	std::uniform_real_distribution<float> d_x(min.x, max.x);
	std::uniform_real_distribution<float> d_y(min.y, max.y);
	std::uniform_real_distribution<float> d_z(min.z, max.z);

	for (size_t i = 0; i < num; i++) {
		particles.push_back(d_x(generator));
		particles.push_back(d_y(generator));
		particles.push_back(d_z(generator));
	}
}


void FluidClothScene1::AddBoundaryMarkers(std::vector<float>& boundary_particles, const Vector3& bmin, const Vector3& bmax)
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
