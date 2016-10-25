#include "FluidSSRenderer.h"
#include <ncltech\SceneManager.h>
#include <ncltech\NCLDebug.h>
#include <assert.h>

FluidSSRenderer::FluidSSRenderer(const std::string& name)
	: PostProcessEffect()
	, m_glArray(NULL)
	, m_glVerts(NULL)
	, m_glVel(NULL)
	, m_glDensity(NULL)
	, m_glPressure(NULL)
	, m_glBoundaryArray(NULL)
	, m_glBoundaryVerts(NULL)
	, m_shdrParticlesDens(NULL)
	, m_shdrParticlesSimple(NULL)
	, m_shdrParticleAO(NULL)
	, m_shdrFinalDisplay(NULL)
	, m_shdrComputeNormals(NULL)
	, m_shdrBilateralBlur(NULL)
	, m_shdrCurveFlowBlur(NULL)
	, m_shdrGaussianBlur(NULL)
	, m_shdrCopyWithDepth(NULL)
	, m_shdrParticlesNrmDpth(NULL)
	, m_shdrGaussianBlurDepth(NULL)
	, m_MaxParticles(0)
	, m_NumParticles(0)
	, m_NumBoundaryParticles(0)
	, m_Vertices(NULL)
	, m_fboParticle(0)
	, m_texParticleNormals(0)
	, m_texFinal(0)
	, m_texParticleAbsorbtion()
	, m_texParticleDepth()
	, m_BackgroundSceneCubeMap(0)
	, m_RenderAsSpheres(true)
	, m_RenderBoundarySpheres(false)
	, m_SurfaceBlurMode(SURFACE_BLUR_BILATERAL)
	, m_Width(0)
	, m_Height(0)
{
	m_ParticleRadius = 0.1235f;// sqrt(0.4f * 0.4f + 0.4f * 0.4f + 0.4f * 0.4f);
}

FluidSSRenderer::~FluidSSRenderer()
{
	m_MaxParticles = 0;
	m_NumParticles = 0;

	m_BackgroundSceneCubeMap = 0;

	if (m_Vertices)
	{
		delete[] m_Vertices;
		m_Vertices = NULL;
	}

	if (m_glArray)
	{
		glDeleteBuffers(1, &m_glVerts);
		glDeleteBuffers(1, &m_glVel);
		glDeleteBuffers(1, &m_glDensity);
		glDeleteBuffers(1, &m_glPressure);
		glDeleteVertexArrays(1, &m_glArray);
		m_glArray = NULL;
	}

	if (m_glBoundaryArray)
	{
		glDeleteBuffers(1, &m_glBoundaryVerts);
		glDeleteVertexArrays(1, &m_glBoundaryArray);
	}

	if (m_shdrParticlesSimple)
	{
		delete m_shdrParticlesSimple;
		m_shdrParticlesSimple = NULL;

		delete m_shdrParticlesDens;
		m_shdrParticlesDens = NULL;

		delete m_shdrFinalDisplay;
		m_shdrFinalDisplay = NULL;

		delete m_shdrCopyWithDepth;
		m_shdrCopyWithDepth = NULL;

		delete m_shdrComputeNormals;
		m_shdrComputeNormals = NULL;

		delete m_shdrBilateralBlur;
		m_shdrBilateralBlur = NULL;


		delete m_shdrCurveFlowBlur;
		m_shdrCurveFlowBlur = NULL;

		delete m_shdrGaussianBlur;
		m_shdrGaussianBlur = NULL;

		delete m_shdrParticleAO;
		m_shdrParticleAO = NULL;

		delete m_shdrParticlesNrmDpth;
		m_shdrParticlesNrmDpth = NULL;

		delete m_shdrGaussianBlurDepth;
		m_shdrGaussianBlurDepth = NULL;
	}

	if (m_fboParticle)
	{
		glDeleteFramebuffers(1, &m_fboParticle);
		glDeleteTextures(1, &m_texParticleNormals);
		glDeleteTextures(1, &m_texFinal);
		m_texParticleDepth.Release();
		m_texParticleAbsorbtion.Release();
	}
}

void FluidSSRenderer::SetWorldSpace(const Vector3& world_origin, const Vector3& sim_ratio)
{
	m_ModelMatrix = Matrix4::Scale(sim_ratio);
	m_ModelMatrix.SetPositionVector(world_origin);
}

void FluidSSRenderer::BuildBuffers(int max_particles)
{
	if (max_particles == m_MaxParticles)
		return;

	m_MaxParticles = max_particles;

	m_Vertices = new Vector3[m_MaxParticles];

	if (!m_glArray) glGenVertexArrays(1, &m_glArray);
	glBindVertexArray(m_glArray);

	//Buffer vertex data	
	if (!m_glVerts) glGenBuffers(1, &m_glVerts);
	glBindBuffer(GL_ARRAY_BUFFER, m_glVerts);
	glBufferData(GL_ARRAY_BUFFER, m_MaxParticles*sizeof(Vector3), NULL, GL_STATIC_DRAW);
	glVertexAttribPointer(VERTEX_BUFFER, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(VERTEX_BUFFER);

	if (!m_glVel) glGenBuffers(1, &m_glVel);
	glBindBuffer(GL_ARRAY_BUFFER, m_glVel);
	glBufferData(GL_ARRAY_BUFFER, m_MaxParticles*sizeof(Vector3), NULL, GL_STATIC_DRAW);
	glVertexAttribPointer(NORMAL_BUFFER, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(NORMAL_BUFFER);

	if (!m_glDensity) glGenBuffers(1, &m_glDensity);
	glBindBuffer(GL_ARRAY_BUFFER, m_glDensity);
	glBufferData(GL_ARRAY_BUFFER, m_MaxParticles*sizeof(float), NULL, GL_STATIC_DRAW);
	glVertexAttribPointer(TANGENT_BUFFER, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(TANGENT_BUFFER);

	if (!m_glPressure) glGenBuffers(1, &m_glPressure);
	glBindBuffer(GL_ARRAY_BUFFER, m_glPressure);
	glBufferData(GL_ARRAY_BUFFER, m_MaxParticles*sizeof(float), NULL, GL_STATIC_DRAW);
	glVertexAttribPointer(COLOUR_BUFFER, 1, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(COLOUR_BUFFER);

	glBindVertexArray(0);


}

void FluidSSRenderer::BuildBoundaryBuffers(int max_boundary_particles, const Vector3* positions)
{
	if (m_NumBoundaryParticles == max_boundary_particles)
		return;


	m_NumBoundaryParticles = max_boundary_particles;

	if (!m_glBoundaryArray) glGenVertexArrays(1, &m_glBoundaryArray);
	glBindVertexArray(m_glBoundaryArray);

	//Buffer vertex data	
	if (!m_glBoundaryVerts) glGenBuffers(1, &m_glBoundaryVerts);
	glBindBuffer(GL_ARRAY_BUFFER, m_glBoundaryVerts);
	glBufferData(GL_ARRAY_BUFFER, m_NumBoundaryParticles*sizeof(Vector3), positions, GL_STATIC_DRAW);
	glVertexAttribPointer(VERTEX_BUFFER, 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(VERTEX_BUFFER);

	glBindVertexArray(0);
}

void FluidSSRenderer::Initialize()
{
	ReloadShaders();
}


bool FluidSSRenderer::OnRender(GLuint depth, GLuint in_col, GLuint out_col)
{
	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_X))
	{
		ReloadShaders();
	}

	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_U))
	{
		m_RenderAsSpheres = !m_RenderAsSpheres;
	}

	if (m_NumParticles == 0)
		return false;


	BuildFBO();

	/*if (m_RenderBoundarySpheres)
		RenderBoundaryBalls(depth, in_col, out_col);*/

	if (m_RenderAsSpheres)
		RenderAsBalls(depth, in_col, out_col);
	else
		RenderAsFluid(depth, in_col, out_col);
	

	return true;



	/*glClearColor(0.6f, 0.6f, 0.6f, 1.0f);

	if (m_Scene->GetCurrentShader() != NULL)
	glUseProgram(m_Scene->GetCurrentShader()->GetProgram());
	else
	glUseProgram(0);*/
}

void BuildTextureComponent(GLuint texture, GLint fbo_component, GLint internalFormat, bool linear, bool repeat, int width, int height) {
	if (texture == NULL)
		std::exception("ERROR: ATTEMPTING TO ALTER NULL TEXTURE!\n");

	glBindTexture(GL_TEXTURE_2D, texture);
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0,
		(fbo_component == GL_DEPTH_ATTACHMENT) ? GL_DEPTH_COMPONENT : GL_RGB,
		(fbo_component == GL_DEPTH_ATTACHMENT) ? GL_FLOAT : GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linear ? GL_LINEAR : GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linear ? GL_LINEAR : GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);

	if (fbo_component == GL_DEPTH_ATTACHMENT) {

		//Has to be GL_NONE in order to correctly sample from glsl shader, 
		//	enable again to allow for hardware optimisation (but without the chance to correctly sample manually)
		//glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);


		//Handle depth as a normal texture/sampler2D
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	}

	glBindTexture(GL_TEXTURE_2D, 0);
}

void BuildTextureComponent(GLuint* texture, GLint fbo_component, GLint internalFormat, bool linear, bool repeat, int width, int height) {
	if (*texture == NULL)
		glGenTextures(1, texture);

	glBindTexture(GL_TEXTURE_2D, *texture);
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0,
		(fbo_component == GL_DEPTH_ATTACHMENT) ? GL_DEPTH_COMPONENT : GL_RGB,
		(fbo_component == GL_DEPTH_ATTACHMENT) ? GL_FLOAT : GL_UNSIGNED_BYTE, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, linear ? GL_LINEAR : GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, linear ? GL_LINEAR : GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeat ? GL_REPEAT : GL_CLAMP_TO_EDGE);

	if (fbo_component == GL_DEPTH_ATTACHMENT) {

		//Has to be GL_NONE in order to correctly sample from glsl shader, 
		//	enable again to allow for hardware optimisation (but without the chance to correctly sample manually)
		//glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
		//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE);


		//Handle depth as a normal texture/sampler2D
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
	}

	glBindTexture(GL_TEXTURE_2D, 0);
}

void FluidSSRenderer::BuildFBO()
{
	int width = SceneManager::Instance()->GetRenderWidth();
	int height = SceneManager::Instance()->GetRenderHeight();

	if (m_Width != width || m_Height != height)
	{
		m_Width = width;
		m_Height = height;

		m_texParticleAbsorbtion.Generate();
		m_texParticleDepth.Generate();

		BuildTextureComponent(&m_texFinal, GL_COLOR_ATTACHMENT0, GL_RGBA8, false, false, width, height);
		BuildTextureComponent(&m_texParticleNormals, GL_COLOR_ATTACHMENT0, GL_RGBA16F, false, false, width, height);
		BuildTextureComponent(m_texParticleAbsorbtion.GetCurrent(), GL_COLOR_ATTACHMENT0, GL_R32F, false, false, width, height);
		BuildTextureComponent(m_texParticleAbsorbtion.GetNext(), GL_COLOR_ATTACHMENT0, GL_R32F, false, false, width, height);
		BuildTextureComponent(m_texParticleDepth.GetCurrent(), GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT32, false, false, width, height);
		BuildTextureComponent(m_texParticleDepth.GetNext(), GL_DEPTH_ATTACHMENT, GL_DEPTH_COMPONENT32, false, false, width, height);


		if (m_fboParticle == NULL) {
			glGenFramebuffers(1, &m_fboParticle);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, m_fboParticle);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_texParticleDepth.GetCurrent(), 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texParticleNormals, 0);

		GLenum buffers[] = { GL_COLOR_ATTACHMENT0 };
		glDrawBuffers(1, buffers);


		GLenum status;
		status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		if (status != GL_FRAMEBUFFER_COMPLETE) {
			printf("FRAMEBUFFER ERROR: %d\n", status);
			assert(false);
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

void FluidSSRenderer::ReloadShaders()
{
	if (m_shdrParticlesSimple) delete m_shdrParticlesSimple;
	if (m_shdrParticlesDens) delete m_shdrParticlesDens;
	if (m_shdrComputeNormals) delete m_shdrComputeNormals;
	if (m_shdrBilateralBlur) delete m_shdrBilateralBlur;
	if (m_shdrCurveFlowBlur) delete m_shdrCurveFlowBlur;
	if (m_shdrGaussianBlur) delete m_shdrGaussianBlur;
	if (m_shdrFinalDisplay) delete m_shdrFinalDisplay;
	if (m_shdrCopyWithDepth) delete m_shdrCopyWithDepth;
	if (m_shdrParticleAO) delete m_shdrParticleAO;
	if (m_shdrParticlesNrmDpth) delete m_shdrParticlesNrmDpth;

	m_shdrParticlesSimple = new Shader(SHADERDIR"fluid/Particle.vert", SHADERDIR"fluid/Particle.frag", SHADERDIR"fluid/Particle.geom");
	m_shdrParticlesSimple->LinkProgram();

	m_shdrParticlesNrmDpth = new Shader(SHADERDIR"fluid/Particle.vert", SHADERDIR"fluid/ParticleNrmDpth.frag", SHADERDIR"fluid/Particle.geom");
	m_shdrParticlesNrmDpth->LinkProgram();

	m_shdrParticlesDens = new Shader(SHADERDIR"fluid/ParticleDensity.vert", SHADERDIR"fluid/ParticleDensity.frag", SHADERDIR"fluid/ParticleDensity.geom");
	m_shdrParticlesDens->LinkProgram();

	m_shdrParticleAO = new Shader(SHADERDIR"fluid/Particle.vert", SHADERDIR"fluid/ParticleAO.frag", SHADERDIR"fluid/Particle.geom");
	m_shdrParticleAO->LinkProgram();

	m_shdrComputeNormals = new Shader(SHADERDIR"fluid/FullScreen.vert", SHADERDIR"fluid/ComputeNormals.frag", SHADERDIR"fluid/FullScreen.geom");
	m_shdrComputeNormals->LinkProgram();

	m_shdrBilateralBlur = new Shader(SHADERDIR"fluid/FullScreen.vert", SHADERDIR"fluid/BilateralFilter.frag", SHADERDIR"fluid/FullScreen.geom");
	m_shdrBilateralBlur->LinkProgram();

	m_shdrCurveFlowBlur = new Shader(SHADERDIR"fluid/FullScreen.vert", SHADERDIR"fluid/CurveFlowFilter.frag", SHADERDIR"fluid/FullScreen.geom");
	m_shdrCurveFlowBlur->LinkProgram();

	m_shdrGaussianBlur = new Shader(SHADERDIR"fluid/FullScreen.vert", SHADERDIR"fluid/GaussianFilter.frag", SHADERDIR"fluid/FullScreen.geom");
	m_shdrGaussianBlur->LinkProgram();

	m_shdrGaussianBlurDepth = new Shader(SHADERDIR"fluid/FullScreen.vert", SHADERDIR"fluid/GaussianFilterDepth.frag", SHADERDIR"fluid/FullScreen.geom");
	m_shdrGaussianBlurDepth->LinkProgram();

	m_shdrFinalDisplay = new Shader(SHADERDIR"fluid/FullScreen.vert", SHADERDIR"fluid/FullScreen.frag", SHADERDIR"fluid/FullScreen.geom");
	m_shdrFinalDisplay->LinkProgram();

	m_shdrCopyWithDepth = new Shader(SHADERDIR"fluid/FullScreen.vert", SHADERDIR"fluid/CopyWithDepth.frag", SHADERDIR"fluid/FullScreen.geom");
	m_shdrCopyWithDepth->LinkProgram();
}


void FluidSSRenderer::RenderToDepthAndThickness(GLuint depth)
{
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texParticleAbsorbtion.GetNext(), 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
	glClear(GL_COLOR_BUFFER_BIT);



	//Render Accumulative Thickness
	glUseProgram(m_shdrParticlesSimple->GetProgram());
	glUniform1f(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "sphereRadius"), 1.5 * m_ParticleRadius);
	glUniform1f(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "sphereAlpha"), 0.001f);
	glUniform1f(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "densityCutOff"), 0.3f);


	glDepthMask(GL_FALSE);
	glBlendFunc(GL_ONE, GL_ONE);
	glDrawArrays(GL_POINTS, 0, m_NumParticles);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDepthMask(GL_TRUE);


	//Render Depth
	glUniform1f(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "densityCutOff"), 0.0f);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texParticleNormals, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_texParticleDepth.GetNext(), 0);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	glDrawBuffers(0, GL_NONE);
	glDrawArrays(GL_POINTS, 0, m_NumParticles);

	m_texParticleAbsorbtion.Swap();
	m_texParticleDepth.Swap();
}

void FluidSSRenderer::BilateralBlurPass()
{
	//Blur the depth buffer
	//glDepthMask(GL_FALSE);
	//GLenum buffers[] = { GL_COLOR_ATTACHMENT0 }; glDrawBuffers(0, buffers);
	glDrawBuffers(0, GL_NONE);

	glUseProgram(m_shdrBilateralBlur->GetProgram());
	glUniform1i(glGetUniformLocation(m_shdrBilateralBlur->GetProgram(), "texDepth"), 0);
	glUniform1i(glGetUniformLocation(m_shdrBilateralBlur->GetProgram(), "texAbsorb"), 1);

	//Bind Depth Texture #1 (Original)
	//GLenum buffers[] = { GL_COLOR_ATTACHMENT0 };
	//glDrawBuffers(1, buffers);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_texParticleDepth.GetNext(), 0);
	//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texParticleAbsorbtion.GetNext(), 0);
	//glClear(GL_DEPTH_BUFFER_BIT);


	glActiveTexture(GL_TEXTURE0);  glBindTexture(GL_TEXTURE_2D, m_texParticleDepth.GetCurrent());
	glActiveTexture(GL_TEXTURE1);  glBindTexture(GL_TEXTURE_2D, m_texParticleAbsorbtion.GetCurrent());
	//glDepthFunc(GL_ALWAYS);
	glDrawArrays(GL_POINTS, 0, 1);
	//glDepthFunc(GL_LEQUAL);

	//m_texParticleDepth.Swap();
	//m_texParticleAbsorbtion.Swap();
}

void FluidSSRenderer::CurveFlowPass()
{
	//Blur the depth buffer
	//glDepthMask(GL_FALSE);
	//GLenum buffers[] = { GL_COLOR_ATTACHMENT0 }; glDrawBuffers(0, buffers);
	glDrawBuffers(0, GL_NONE);

	glUseProgram(m_shdrCurveFlowBlur->GetProgram());
	glUniform1i(glGetUniformLocation(m_shdrCurveFlowBlur->GetProgram(), "particleTexture"), 0);

	//Bind Depth Texture #1 (Original)
	//GLenum buffers[] = { GL_COLOR_ATTACHMENT0 };
	//glDrawBuffers(1, buffers);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_texParticleDepth.GetNext(), 0);
	//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texParticleAbsorbtion.GetNext(), 0);
	//glClear(GL_DEPTH_BUFFER_BIT);


	glActiveTexture(GL_TEXTURE0);  glBindTexture(GL_TEXTURE_2D, m_texParticleDepth.GetCurrent());
	//glDepthFunc(GL_ALWAYS);
	glDrawArrays(GL_POINTS, 0, 1);
	//glDepthFunc(GL_LEQUAL);

	//m_texParticleDepth.Swap();
	//m_texParticleAbsorbtion.Swap();
}

void FluidSSRenderer::GaussianFilter(PingPongTex& tex, bool isDepth)
{
	Shader* shdr;
	if (isDepth)
	{
		glDrawBuffers(0, GL_NONE);
		shdr = m_shdrGaussianBlurDepth;
	}
	else
	{
		GLenum buffers[] = { GL_COLOR_ATTACHMENT0 }; glDrawBuffers(1, buffers);
		shdr = m_shdrGaussianBlur;
		
	}

	glUseProgram(shdr->GetProgram());
	glUniform1i(glGetUniformLocation(shdr->GetProgram(), "particleTexture"), 0);

	glActiveTexture(GL_TEXTURE0);

	if (!isDepth) glDepthMask(GL_FALSE);

	glUniform2f(glGetUniformLocation(shdr->GetProgram(), "blurDirection"), 1.0f, 0.0f);
	if (isDepth)
	{
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex.GetNext(), 0);
		glClear(GL_DEPTH_BUFFER_BIT);
	}	
	else
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.GetNext(), 0);
	
	glBindTexture(GL_TEXTURE_2D, tex.GetCurrent());

	glDrawArrays(GL_POINTS, 0, 1);
	tex.Swap();

	glUniform2f(glGetUniformLocation(shdr->GetProgram(), "blurDirection"), 0.0f, 1.0f);
	if (isDepth) 
	{
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, tex.GetNext(), 0);
		glClear(GL_DEPTH_BUFFER_BIT);
	}
	else
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex.GetNext(), 0);

	glBindTexture(GL_TEXTURE_2D, tex.GetCurrent());

	glDrawArrays(GL_POINTS, 0, 1);
	tex.Swap();

	if (!isDepth) glDepthMask(GL_TRUE);
}

int iterations = 40;
void FluidSSRenderer::RenderAsFluid(GLuint depth, GLuint in_col, GLuint out_col)
{
#define BLUR_ONE_PASS TRUE
	Matrix4 projMatrix = SceneManager::Instance()->GetProjMatrix();
	Matrix4 viewMatrix = SceneManager::Instance()->GetViewMatrix();
	Matrix4 invProjMatrix = Matrix4::Inverse(projMatrix);
	Matrix4 invProjViewMatrix = Matrix4::Inverse(projMatrix * viewMatrix);

	float width = (float)SceneManager::Instance()->GetRenderWidth();
	float height = (float)SceneManager::Instance()->GetRenderHeight();
	Vector2 pixel = Vector2(1.0f / width, 1.0f / height);


	glUseProgram(m_shdrParticlesSimple->GetProgram());
	glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "mdlMatrix"), 1, false, (float*)&m_ModelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "viewMatrix"), 1, false, (float*)&viewMatrix);
	glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "projMatrix"), 1, false, (float*)&projMatrix);

	glUseProgram(m_shdrBilateralBlur->GetProgram());
	glUniformMatrix4fv(glGetUniformLocation(m_shdrBilateralBlur->GetProgram(), "projMatrix"), 1, false, (float*)&projMatrix);
	glUniformMatrix4fv(glGetUniformLocation(m_shdrBilateralBlur->GetProgram(), "invProjMatrix"), 1, false, (float*)&invProjMatrix);
	glUniform2f(glGetUniformLocation(m_shdrBilateralBlur->GetProgram(), "screen_size"), width, height);

	glUseProgram(m_shdrComputeNormals->GetProgram());
	glUniformMatrix4fv(glGetUniformLocation(m_shdrComputeNormals->GetProgram(), "invProjMatrix"), 1, false, (float*)&invProjViewMatrix);
	glUniform2fv(glGetUniformLocation(m_shdrComputeNormals->GetProgram(), "texelSize"), 1, &pixel.x);

	glUseProgram(m_shdrCurveFlowBlur->GetProgram());
	glUniform2f(glGetUniformLocation(m_shdrCurveFlowBlur->GetProgram(), "screenSize"), width, height);
	glUniformMatrix4fv(glGetUniformLocation(m_shdrCurveFlowBlur->GetProgram(), "projection"), 1, false, (float*)&projMatrix);

	glUseProgram(m_shdrGaussianBlur->GetProgram());
	glUniform2f(glGetUniformLocation(m_shdrGaussianBlur->GetProgram(), "screenSize"), width, height);

	glUseProgram(m_shdrGaussianBlurDepth->GetProgram());
	glUniform2f(glGetUniformLocation(m_shdrGaussianBlurDepth->GetProgram(), "screenSize"), width, height);

	glBindVertexArray(m_glArray);

	glBindFramebuffer(GL_FRAMEBUFFER, m_fboParticle);

	GLenum drawbuffers[] = { GL_COLOR_ATTACHMENT0 };
	glDrawBuffers(1, drawbuffers);
	glViewport(0, 0, SceneManager::Instance()->GetRenderWidth(), SceneManager::Instance()->GetRenderHeight());
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);


	RenderToDepthAndThickness(depth);



	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_texParticleDepth.GetNext(), 0);
	glClear(GL_DEPTH_BUFFER_BIT);
	glDepthFunc(GL_ALWAYS);

	for (int i = 0; i < 10; ++i)
		GaussianFilter(m_texParticleAbsorbtion, false);

	for (int i = 0; i < 4; ++i)
		GaussianFilter(m_texParticleDepth, true);

	glDrawBuffers(1, drawbuffers);
	//BilateralBlurPass();
	//m_texParticleDepth.Swap();

	for (int i = 0; i < iterations; ++i)
	{
		CurveFlowPass();
		m_texParticleDepth.Swap();
	}


	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_UP))
	{
		iterations += 5;
	}

	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_DOWN))
	{
		iterations -= 5;
		if (iterations < 5) iterations = 0;
	}

	NCLDebug::AddStatusEntry(Vector4(1.0f, 1.0f, 1.0f, 1.0f), "Blur Iterations: %d\n", iterations);


	glDrawBuffers(1, drawbuffers);

	/*
	//Blur the depth buffer
	//glDepthMask(GL_FALSE);
	GLenum buffers[] = { GL_COLOR_ATTACHMENT0 }; glDrawBuffers(1, buffers);


	glUseProgram(m_shdrBilateralBlur->GetProgram());
	glUniform1i(glGetUniformLocation(m_shdrBilateralBlur->GetProgram(), "texDepth"), 0);
	glUniform1i(glGetUniformLocation(m_shdrBilateralBlur->GetProgram(), "texAbsorb"), 1);

	//Bind Depth Texture #1 (Original)
	//GLenum buffers[] = { GL_COLOR_ATTACHMENT0 };
	//glDrawBuffers(1, buffers);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_texParticleDepth.GetNext(), 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texParticleAbsorbtion.GetNext(), 0);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);


	glActiveTexture(GL_TEXTURE0);  glBindTexture(GL_TEXTURE_2D, m_texParticleDepth.GetCurrent());
	glActiveTexture(GL_TEXTURE1);  glBindTexture(GL_TEXTURE_2D, m_texParticleAbsorbtion.GetCurrent());
	//glDepthFunc(GL_ALWAYS);
	glDrawArrays(GL_POINTS, 0, 1);
	//glDepthFunc(GL_LEQUAL);

	m_texParticleDepth.Swap();
	m_texParticleAbsorbtion.Swap();
	*/










	//Compute surface normals
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texParticleNormals, 0);


	glUseProgram(m_shdrComputeNormals->GetProgram());
	glUniform1i(glGetUniformLocation(m_shdrComputeNormals->GetProgram(), "texDepth"), 1);

	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, m_texParticleDepth.GetCurrent());

	glDepthMask(GL_FALSE);
	glClear(GL_COLOR_BUFFER_BIT);
	glDrawArrays(GL_POINTS, 0, 1);














	//Render Fullscreen Quad

	glDepthFunc(GL_LEQUAL);
	glDepthMask(GL_TRUE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out_col, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
	//glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	//glBindFramebuffer(GL_FRAMEBUFFER, m_Scene->m_ScreenFBO);
	//glDisable(GL_CULL_FACE);
	Vector3 camPos = SceneManager::Instance()->GetCamera()->GetPosition();

	Matrix3 viewMatrix3 = Matrix3(viewMatrix);
	glUseProgram(m_shdrFinalDisplay->GetProgram());
	glUniformMatrix4fv(glGetUniformLocation(m_shdrFinalDisplay->GetProgram(), "invProjViewMatrix"), 1, false, (float*)&invProjViewMatrix);
	glUniformMatrix3fv(glGetUniformLocation(m_shdrFinalDisplay->GetProgram(), "viewMatrix"), 1, false, (float*)&viewMatrix3);
	glUniform3fv(glGetUniformLocation(m_shdrFinalDisplay->GetProgram(), "ambientColour"), 1, &SceneManager::Instance()->GetAmbientColor().x);
	glUniform3fv(glGetUniformLocation(m_shdrFinalDisplay->GetProgram(), "invLightDir"), 1, &SceneManager::Instance()->GetInverseLightDirection().x);
	//glUniform1f(glGetUniformLocation(m_shdrFinalDisplay->GetProgram(), "specularIntensity"), SceneManager::Instance()->GetSpecularIntensity());
	glUniform3fv(glGetUniformLocation(m_shdrFinalDisplay->GetProgram(), "cameraPos"), 1, &camPos.x);
	glUniform1i(glGetUniformLocation(m_shdrFinalDisplay->GetProgram(), "texNormals"), 0);
	glUniform1i(glGetUniformLocation(m_shdrFinalDisplay->GetProgram(), "texAbsorbtion"), 1);
	glUniform1i(glGetUniformLocation(m_shdrFinalDisplay->GetProgram(), "texDepth"), 2);
	glUniform1i(glGetUniformLocation(m_shdrFinalDisplay->GetProgram(), "texReflect"), 3);
	glUniform1i(glGetUniformLocation(m_shdrFinalDisplay->GetProgram(), "texRefract"), 4);

	glActiveTexture(GL_TEXTURE4); glBindTexture(GL_TEXTURE_2D, in_col);
	glActiveTexture(GL_TEXTURE3); glBindTexture(GL_TEXTURE_CUBE_MAP, m_BackgroundSceneCubeMap);
	glActiveTexture(GL_TEXTURE2); glBindTexture(GL_TEXTURE_2D, m_texParticleDepth.GetCurrent());
	glActiveTexture(GL_TEXTURE1); glBindTexture(GL_TEXTURE_2D, m_texParticleAbsorbtion.GetCurrent());
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, m_texParticleNormals);

	glDrawArrays(GL_POINTS, 0, 1);


	/*glUseProgram(m_shdrCopyWithDepth->GetProgram());
	glUniform1i(glGetUniformLocation(m_shdrCopyWithDepth->GetProgram(), "texColor"), 0);
	glUniform1i(glGetUniformLocation(m_shdrCopyWithDepth->GetProgram(), "texDepth"), 2);
	glActiveTexture(GL_TEXTURE0); glBindTexture(GL_TEXTURE_2D, m_texFinal);



	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out_col, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
	glDrawArrays(GL_POINTS, 0, 1);*/
	//glBindVertexArray(0);*/


}


bool ambient_occlusion = false;
void FluidSSRenderer::RenderAsBalls(GLuint depth, GLuint in_col, GLuint out_col)
{
	const float AO_Factor = 2.0f; //x times the radius of the particles
	if (Window::GetKeyboard()->KeyTriggered(KEYBOARD_O))
		ambient_occlusion = !ambient_occlusion;


	Matrix4 projMatrix = SceneManager::Instance()->GetProjMatrix();
	Matrix4 viewMatrix = SceneManager::Instance()->GetViewMatrix();
	Matrix4 invProjMatrix = Matrix4::Inverse(projMatrix);
	Matrix4 invProjViewMatrix = Matrix4::Inverse(projMatrix * viewMatrix);

	glBindVertexArray(m_glArray);
	GLenum drawbuffers[] = { GL_COLOR_ATTACHMENT0 };

	if (ambient_occlusion)
	{
		//First Render to Depth Texture
		glBindFramebuffer(GL_FRAMEBUFFER, m_fboParticle);
		glViewport(0, 0, SceneManager::Instance()->GetRenderWidth(), SceneManager::Instance()->GetRenderHeight());
		glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

		glDepthMask(GL_TRUE);

		//Render Depth+ Normals
		glUseProgram(m_shdrParticlesNrmDpth->GetProgram());
		glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesNrmDpth->GetProgram(), "mdlMatrix"), 1, false, (float*)&m_ModelMatrix);
		glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesNrmDpth->GetProgram(), "viewMatrix"), 1, false, (float*)&viewMatrix);
		glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesNrmDpth->GetProgram(), "projMatrix"), 1, false, (float*)&projMatrix);
		glUniform1f(glGetUniformLocation(m_shdrParticlesNrmDpth->GetProgram(), "sphereRadius"), m_ParticleRadius);
		glUniform1f(glGetUniformLocation(m_shdrParticlesNrmDpth->GetProgram(), "sphereAlpha"), 1.0f);
		glUniform1f(glGetUniformLocation(m_shdrParticlesNrmDpth->GetProgram(), "densityCutOff"), 0.0f);

		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texParticleNormals, 0);

		
		glDrawBuffers(1, drawbuffers);
		glClear(GL_COLOR_BUFFER_BIT);


		glDrawArrays(GL_POINTS, 0, m_NumParticles);

		m_texParticleDepth.Swap();



		//Next Approximate Ambient Occlusion
		float tanHalfFOV = 1.0 / projMatrix[5];

		glUseProgram(m_shdrParticleAO->GetProgram());
		glUniformMatrix4fv(glGetUniformLocation(m_shdrParticleAO->GetProgram(), "mdlMatrix"), 1, false, (float*)&m_ModelMatrix);
		glUniformMatrix4fv(glGetUniformLocation(m_shdrParticleAO->GetProgram(), "viewMatrix"), 1, false, (float*)&viewMatrix);
		glUniformMatrix4fv(glGetUniformLocation(m_shdrParticleAO->GetProgram(), "projMatrix"), 1, false, (float*)&projMatrix);
		glUniform1f(glGetUniformLocation(m_shdrParticleAO->GetProgram(), "sphereRadiusNrml"), m_ParticleRadius);
		glUniform1f(glGetUniformLocation(m_shdrParticleAO->GetProgram(), "sphereRadius"), AO_Factor * m_ParticleRadius);
		glUniform2f(glGetUniformLocation(m_shdrParticleAO->GetProgram(), "screenSize"), SceneManager::Instance()->GetRenderWidth(), SceneManager::Instance()->GetRenderHeight());
		glUniform1f(glGetUniformLocation(m_shdrParticleAO->GetProgram(), "tanHalfFOV"), tanHalfFOV);
		glUniform2f(glGetUniformLocation(m_shdrParticleAO->GetProgram(), "projNearFar"), 0.01f, 1000.0f);
		glUniform1i(glGetUniformLocation(m_shdrParticleAO->GetProgram(), "depthTex"), 5);
		glUniform1i(glGetUniformLocation(m_shdrParticleAO->GetProgram(), "normalTex"), 6);
		glUniform1i(glGetUniformLocation(m_shdrParticleAO->GetProgram(), "densityCutOff"), 0.0f);
		glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, m_texParticleDepth.GetCurrent());
		glActiveTexture(GL_TEXTURE6); glBindTexture(GL_TEXTURE_2D, m_texParticleNormals);


		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_texParticleAbsorbtion.GetNext(), 0);
		glDrawBuffers(1, drawbuffers);
		glClear(GL_COLOR_BUFFER_BIT);

		glDepthMask(GL_FALSE);
		glBlendFunc(GL_ONE, GL_ONE);
		glDrawArrays(GL_POINTS, 0, m_NumParticles);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glDepthMask(GL_TRUE);
	}

	const Vector3 eyeLightDir = -SceneManager::Instance()->GetInverseLightDirection();// Vector3(0.9f, -1.0f, 0.5f).Normalise();
	Vector3 wsLightDir = Matrix3(viewMatrix) * eyeLightDir;

	glUseProgram(m_shdrParticlesDens->GetProgram());
	glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesDens->GetProgram(), "mdlMatrix"), 1, false, (float*)&m_ModelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesDens->GetProgram(), "viewMatrix"), 1, false, (float*)&viewMatrix);
	glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesDens->GetProgram(), "projMatrix"), 1, false, (float*)&projMatrix);
	glUniform3fv(glGetUniformLocation(m_shdrParticlesDens->GetProgram(), "lightDir"), 1, (float*)&wsLightDir.x);
	glUniform1f(glGetUniformLocation(m_shdrParticlesDens->GetProgram(), "sphereRadius"), m_ParticleRadius);
	glUniform1i(glGetUniformLocation(m_shdrParticlesDens->GetProgram(), "aoTex"), 5);
	glUniform2f(glGetUniformLocation(m_shdrParticlesDens->GetProgram(), "screenSize"), SceneManager::Instance()->GetRenderWidth(), SceneManager::Instance()->GetRenderHeight());
	glActiveTexture(GL_TEXTURE5); glBindTexture(GL_TEXTURE_2D, m_texParticleAbsorbtion.GetNext());
	glUniform1i(glGetUniformLocation(m_shdrParticlesDens->GetProgram(), "useAO"), ambient_occlusion ? 1 : 0);

	glBindFramebuffer(GL_FRAMEBUFFER, m_fboParticle);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depth, 0);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out_col, 0);
	glDrawBuffers(1, drawbuffers);
	glDrawArrays(GL_POINTS, 0, m_NumParticles);
}

void FluidSSRenderer::RenderBoundaryBalls(GLuint depth, GLuint in_col, GLuint out_col)
{
	Matrix4 projMatrix = SceneManager::Instance()->GetProjMatrix();
	Matrix4 viewMatrix = SceneManager::Instance()->GetViewMatrix();
	Matrix4 invProjMatrix = Matrix4::Inverse(projMatrix);
	Matrix4 invProjViewMatrix = Matrix4::Inverse(projMatrix * viewMatrix);

	glBindVertexArray(m_glBoundaryArray);
	glDepthMask(GL_TRUE);
	const Vector3 eyeLightDir = Vector3(0.9f, -1.0f, 0.5f).Normalise();

	Vector3 wsLightDir = Matrix3(viewMatrix) * eyeLightDir;

	//First draw the particles again, building up an eye space fluid depth map
	glUseProgram(m_shdrParticlesSimple->GetProgram());
	glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "mdlMatrix"), 1, false, (float*)&m_ModelMatrix);
	glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "viewMatrix"), 1, false, (float*)&viewMatrix);
	glUniformMatrix4fv(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "projMatrix"), 1, false, (float*)&projMatrix);
	glUniform3fv(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "lightDir"), 1, (float*)&wsLightDir.x);
	glUniform1f(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "sphereRadius"), m_ParticleRadius);
	glUniform1f(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "sphereAlpha"), -1.0f);
	glUniform3f(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "color"), 1.0f, 0.7f, 0.2f);
	glUniform1f(glGetUniformLocation(m_shdrParticlesSimple->GetProgram(), "densityCutOff"), -1.0f);

	glDrawArrays(GL_POINTS, 0, m_NumBoundaryParticles);
	glBindVertexArray(0);
}