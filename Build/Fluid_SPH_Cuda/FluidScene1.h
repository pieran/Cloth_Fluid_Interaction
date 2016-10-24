
#pragma once
#include <nclgl\OGLRenderer.h>
#include <nclgl\Mesh.h>
#include <ncltech\Scene.h>
#include <random>

#include "FluidSSRenderer.h"
#include <libsim\PicFlip.cuh>


class FluidScene1 : public Scene
{
public:
	FluidScene1(const std::string& friendly_name);
	virtual ~FluidScene1();

	virtual void OnInitializeScene()	 override;
	virtual void OnCleanupScene()		 override;
	virtual void OnUpdateScene(float dt) override;


protected:
	void AddBoundaryMarkers(std::vector<float>& boundary_particles, const Vector3& bmin, const Vector3& bmax);
	void GenerateFluidCube_Random(std::vector<float>& particles, const Vector3& min, const Vector3& max, size_t num);


protected:
	FluidSSRenderer* fluid_renderer;
	FluidPicFlip* fluid;
	
	std::default_random_engine generator;
};