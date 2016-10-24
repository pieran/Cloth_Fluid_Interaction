#pragma once

#include <nclgl\OGLRenderer.h>
#include <nclgl\Matrix4.h>
#include <ncltech\Object.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>

#include "XPBDConstraints.h"
#include "XPBD_ConstraintLayer.h"





class XPBD : public Object
{
	friend class XPBD_ConstraintLayer;
	friend class FluidClothCoupler;
public:
	XPBD();
	virtual ~XPBD();
	void Release();

	void InitializeCloth(int div_x, int div_y, const Matrix4& transform);

	inline void SetTextureFront(GLuint tex, bool managed = true)
	{
		m_glTexFront = tex;
		m_glManageTexFront = managed;
	}
	inline void SetTextureBack(GLuint tex, bool managed = true)
	{
		m_glTexBack = tex;
		m_glManageTexBack = managed;
	}
	std::vector<XPBDSphereConstraint> m_SphereConstraints;

	void Update(float dt);
	void GenerateNormals(bool reverseOrder);
	void UpdateGLBuffers();

protected:


	void InitializeCuda(const Vector4* initial_positions);
	void InitializeGL(const Vector4* initial_positions);


	virtual void OnRenderObject() override;
	virtual void OnUpdateObject(float dt) override;

	

	
protected:
	//Host
	int m_SolverIterations = 120;
	float m_MaterialBend = 0.1f;
	float m_MaterialStretchWeft = 0.6f;// 0.8f;
	float m_MaterialStretchWarp = 0.7f;// 0.95f;
	float m_MaterialShear = 0.5f;// 0.8f;
	float m_MaterialDamp = 0.01f;


	int m_NumParticles, m_NumIndicies, m_NumConstraintIndicies, m_NumX, m_NumY;

	//Constraints
	std::vector<XPBD_ConstraintLayer> m_ConstraintLayers;
	

	//Device
	CudaTextureWrapper m_cuParticlePosOld;
	CudaTextureWrapper m_cuParticlePos;			//float4 -> [x,y,z,(invmass)]
	CudaTextureWrapper m_cuParticlePosTmp;

	CudaTextureWrapper m_cuParticleFaceNormals;
	CudaTextureWrapper m_cuParticleVertNormals;



	//Rendering
	bool m_glManageTexFront, m_glManageTexBack;
	GLuint m_glTexFront;
	GLuint m_glTexBack;
	GLuint m_glArr;
	GLuint m_glBuffers[5]; //0: Verts, 1: Normals, 2: Tex-Coords, 3: Indicies (tris), 4: Indicies (constraints)

	cudaGraphicsResource_t m_cuVerticesRef;
	cudaGraphicsResource_t m_cuNormalsRef;
	float4* m_cuVerticesBufRef;
};