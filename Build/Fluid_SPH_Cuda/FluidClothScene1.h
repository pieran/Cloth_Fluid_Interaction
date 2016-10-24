
#pragma once
#include <nclgl\OGLRenderer.h>
#include <nclgl\Mesh.h>
#include <ncltech\Scene.h>
#include <random>

#include "FluidSSRenderer.h"
#include <libsim\PicFlip.cuh>
#include <libsim\XPBD.h>
#include "FluidClothCoupler.cuh"

class FluidClothScene1 : public Scene
{
public:
	FluidClothScene1(const std::string& friendly_name);
	virtual ~FluidClothScene1();

	virtual void OnInitializeScene()	 override;
	virtual void OnCleanupScene()		 override;
	virtual void OnUpdateScene(float dt) override;


protected:
	void AddBoundaryMarkers(std::vector<float>& boundary_particles, const Vector3& bmin, const Vector3& bmax);
	void GenerateFluidCube_Random(std::vector<float>& particles, const Vector3& min, const Vector3& max, float density, const Vector3& grid_dim);


protected:
	float m_Timestep;

	FluidSSRenderer* fluid_renderer;
	FluidPicFlip* fluid;

	FluidClothCoupler* coupler;
	
	XPBD* m_Sim;
	GLuint m_ClothTexFront;
	GLuint m_ClothTexBack;

	std::default_random_engine generator;
};