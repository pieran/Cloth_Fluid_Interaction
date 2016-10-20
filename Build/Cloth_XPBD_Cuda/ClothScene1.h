
#pragma once

#include <nclgl\Mesh.h>
#include <ncltech\Scene.h>
#include "XPBD.h"

class ClothScene1 : public Scene
{
public:
	ClothScene1(const std::string& friendly_name);
	virtual ~ClothScene1();

	virtual void OnInitializeScene()	 override;
	virtual void OnCleanupScene()		 override;
	virtual void OnUpdateScene(float dt) override;

protected:
	XPBD* m_Sim;

	GLuint m_ClothTex;

	uint  m_ShadowCycleKey;
	uint  m_SuperSampleCycleKey;
};