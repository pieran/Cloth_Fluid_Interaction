#pragma once

#include <nclgl\OGLRenderer.h>
#include <nclgl\Matrix4.h>
#include <ncltech\Object.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>

struct XPBDDistanceConstraint { uint p1, p2;	float rest_length, k, lamdaij; };
struct XPBDBendingConstraint { uint p1, p2, p3; float rest_length, k, lambdaij; };
struct XPBDSphereConstraint { float radius; Vector3 centre; };

class XPBD : public Object
{
public:
	XPBD();
	virtual ~XPBD();
	void Release();

	void InitializeCloth(int div_x, int div_y, const Matrix4& transform);

	inline void SetTexture(GLuint tex, bool managed = true)
	{
		m_glTex = tex;
		m_glManageTex = managed;
	}

protected:
	void InitializeConstraints(const Vector3* initial_positions);
	void InitializeCuda(const Vector3* initial_positions);
	void InitializeGL(const Vector3* initial_positions);

	XPBDDistanceConstraint BuildDistanceConstraint(const Vector3* positions, uint idxA, uint idxB, float k);
	XPBDBendingConstraint BuildBendingConstraint(const Vector3* positions, uint idxA, uint idxB, uint idxC, float k);

	virtual void OnRenderObject() override;
	virtual void OnUpdateObject(float dt) override;

protected:
	//Host
	uint m_SolverIterations = 120;
	float m_MaterialBend = 0.75f;
	float m_MaterialStretchWeft = 0.75f;
	float m_MaterialStretchWarp = 0.759f;
	float m_MaterialShear = 0.3f;
	float m_MaterialDamp = 0.1f;


	uint m_NumParticles, m_NumIndicies, m_NumX, m_NumY;
	std::vector<XPBDDistanceConstraint> m_DistanceConstraints;
	std::vector<XPBDBendingConstraint> m_BendingConstraints;
	std::vector<XPBDSphereConstraint> m_SphereConstraints;



	//Device
	cudaTextureObject_t m_cuParticlePosTex;
	cudaTextureObject_t m_cuParticlePosTmpTex;
	cudaArray* m_cuParticlePosArr;
	cudaArray* m_cuParticlePosTmpArr;

	XPBDDistanceConstraint* m_cuDistanceConstraints;
	XPBDBendingConstraint* m_cuBendingConstraints;
	XPBDSphereConstraint* m_cuSphereConstraints;


	//Rendering
	bool m_glManageTex;
	GLuint m_glTex;
	GLuint m_glArr;
	GLuint m_glBuffers[4]; //0: Verts, 1: Normals, 2: Tex-Coords, 3: Indicies
};