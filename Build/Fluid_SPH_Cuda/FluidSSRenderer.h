#pragma once
#include <nclgl\OGLRenderer.h>
#include <nclgl\Shader.h>
#include <ncltech\SceneRenderer.h>

enum SurfaceBlurMode
{
	SURFACE_BLUR_GAUSSIAN = 0,
	SURFACE_BLUR_BILATERAL,
	SURFACE_BLUR_CURVEFLOW
};

struct PingPongTex
{
public:
	PingPongTex() : current(false) { tex[0] = NULL; tex[1] = NULL; }

	void Release()
	{
		if (tex[0] != NULL)
		{
			glDeleteTextures(2, tex);
			tex[0] = NULL;
		}
	}

	void Generate()
	{
		glGenTextures(2, tex);
	}

	GLuint GetCurrent()
	{
		return current ? tex[0] : tex[1];
	}

	GLuint GetNext()
	{
		return current ? tex[1] : tex[0];
	}

	void Swap()
	{
		current = !current;
	}

private:
	GLuint tex[2];
	bool current;
};

class FluidSSRenderer : public PostProcessEffect
{
public:
	FluidSSRenderer(const std::string& name = "");
	virtual ~FluidSSRenderer();


	void Initialize();
	virtual bool OnRender(GLuint depth, GLuint in_col, GLuint out_col) override;


	void SetBackgroundSceneCubemap(GLuint texIdx) { m_BackgroundSceneCubeMap = texIdx; }
	void SetWorldSpace(const Vector3& world_origin, const Vector3& sim_ratio);
	void BuildBuffers(int max_particles);
	void BuildBoundaryBuffers(int max_boundary_particles, const Vector3* positions);


	bool& RenderBoundarySpheres() { return m_RenderBoundarySpheres; }
	SurfaceBlurMode& BlurMode() { return m_SurfaceBlurMode; }

	const Matrix4& GetModelMatrix() { return m_ModelMatrix; }
protected:


	void RenderAsFluid(GLuint depth, GLuint in_col, GLuint out_col);
	void RenderAsBalls(GLuint depth, GLuint in_col, GLuint out_col);
	void RenderBoundaryBalls(GLuint depth, GLuint in_col, GLuint out_col);

	void BuildFBO();
	void ReloadShaders();

	void RenderToDepthAndThickness(GLuint depth);
	void BilateralBlurPass();
	void CurveFlowPass();
	void GaussianFilter(PingPongTex& tex, bool isdepth);

public:
	int m_Width, m_Height;

	bool m_RenderBoundarySpheres;
	bool m_RenderAsSpheres;
	float m_ParticleRadius;
	SurfaceBlurMode m_SurfaceBlurMode;

	uint m_MaxParticles, m_NumParticles;
	GLuint m_glArray;
	GLuint m_glVerts;
	GLuint m_glVel;
	GLuint m_glDensity;
	GLuint m_glPressure;

	uint m_NumBoundaryParticles;
	GLuint m_glBoundaryArray;
	GLuint m_glBoundaryVerts;

	Vector3* m_Vertices;
	GLuint m_BackgroundSceneCubeMap;

	Matrix4 m_ModelMatrix;

	Shader* m_shdrParticlesDens;
	Shader* m_shdrParticleAO;
	Shader* m_shdrParticlesSimple;
	Shader* m_shdrParticlesNrmDpth;
	Shader* m_shdrComputeNormals;
	Shader* m_shdrBilateralBlur;
	Shader* m_shdrGaussianBlur;
	Shader* m_shdrGaussianBlurDepth;
	Shader* m_shdrCurveFlowBlur;
	Shader* m_shdrFinalDisplay;
	Shader* m_shdrCopyWithDepth;



	GLuint m_fboParticle;
	GLuint m_texFinal;
	GLuint m_texParticleNormals;
	PingPongTex m_texParticleDepth;
	PingPongTex m_texParticleAbsorbtion;
};