
#pragma once

#include <nclgl\Mesh.h>
#include <ncltech\Scene.h>

class GraphicsScene : public Scene
{
public:
	GraphicsScene(const std::string& friendly_name);
	virtual ~GraphicsScene();

	virtual void OnInitializeScene()	 override;
	virtual void OnCleanupScene()		 override;
	virtual void OnUpdateScene(float dt) override;

protected:
	float m_AccumTime;
	uint  m_ShadowCycleKey;
	uint  m_SuperSampleCycleKey;
};