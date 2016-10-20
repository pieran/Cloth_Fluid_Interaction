#include "SceneManager.h"
#include "NCLDebug.h"
#include "PhysicsEngine.h"

SceneManager::SceneManager() 
	: SceneRenderer()
	, m_SceneIdx(NULL)
{

}

SceneManager::~SceneManager()
{
	m_SceneIdx = 0;
	for (Scene* scene : m_AllScenes)
	{
		if (scene != m_Scene)
		{
			scene->OnCleanupScene();
			delete scene;
		}
	}
	m_AllScenes.clear();
}


void SceneManager::EnqueueScene(Scene* scene)
{
	if (scene == NULL)
	{
		NCLERROR("Attempting to enqueue NULL scene");
		return;
	}

	m_AllScenes.push_back(scene);

	//If this was the first scene, activate it immediately
	if (m_AllScenes.size() == 1)
		JumpToScene(0);
	else
		Window::GetWindow().SetWindowTitle("NCLTech - [%d/%d] %s", m_SceneIdx + 1, m_AllScenes.size(), m_Scene->GetSceneName().c_str());
}

void SceneManager::JumpToScene()
{
	JumpToScene((m_SceneIdx + 1) % m_AllScenes.size());
}

void SceneManager::JumpToScene(int idx)
{
	if (idx < 0 || idx >= (int)m_AllScenes.size())
	{
		NCLERROR("Invalid Scene Index: %d", idx);
		return;
	}

	//Clear up old scene
	if (m_Scene)
	{
		PhysicsEngine::Instance()->RemoveAllPhysicsObjects();

		m_FrameRenderList->RemoveAllObjects();

		for (uint i = 0; i < m_ShadowMapNum; ++i)
			m_ShadowRenderLists[i]->RemoveAllObjects();

		m_Scene->OnCleanupScene();
	}

	m_SceneIdx = idx;
	m_Scene = m_AllScenes[idx];

	//Initialize new scene
	PhysicsEngine::Instance()->SetDefaults();
	InitializeDefaults();
	m_Scene->OnInitializeScene();
	Window::GetWindow().SetWindowTitle("NCLTech - [%d/%d] %s", idx + 1, m_AllScenes.size(), m_Scene->GetSceneName().c_str());
}

void SceneManager::JumpToScene(const std::string& friendly_name)
{
	bool found = false;
	uint idx = 0;
	for (uint i = 0; found == false && i < m_AllScenes.size(); ++i)
	{
		if (m_AllScenes[i]->GetSceneName() == friendly_name)
		{
			found = true;
			idx = i;
			break;
		}
	}

	if (found)
	{
		JumpToScene(idx);
	}
	else
	{
		NCLERROR("Unknown Scene Alias: \"%s\"", friendly_name.c_str());
	}
}