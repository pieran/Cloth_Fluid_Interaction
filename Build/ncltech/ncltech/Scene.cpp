#include "Scene.h"
#include "CommonMeshes.h"
#include "NCLDebug.h"
#include "PhysicsEngine.h"
#include <algorithm>

Scene::Scene(const std::string& friendly_name)
	: m_SceneName(friendly_name)
{	
	m_RootGameObject = new Object("rootNode");
	m_RootGameObject->m_Scene = this;
	m_RootGameObject->SetBoundingRadius(30.0f); //Default scene radius of 40m
}

Scene::~Scene()
{
	if (m_RootGameObject)
	{
		delete m_RootGameObject;
		m_RootGameObject = NULL;
	}
}

void Scene::AddGameObject(Object* game_object)
{
	m_RootGameObject->AddChildObject(game_object);
}

Object* Scene::FindGameObject(const std::string& name)
{
	return m_RootGameObject->FindGameObject(name);
}

void Scene::DeleteAllGameObjects()
{
	if (m_RootGameObject)
	{
		for (auto child : m_RootGameObject->m_Children)
		{
			delete child;
		}
		m_RootGameObject->m_Children.clear();
	}
}

void Scene::OnUpdateScene(float dt)
{
	UpdateNode(dt, m_RootGameObject);
}

void Scene::BuildWorldMatrices()
{
	UpdateWorldMatrices(m_RootGameObject, Matrix4());
}

void Scene::UpdateWorldMatrices(Object* cNode, const Matrix4& parentWM)
{
	if (cNode->HasPhysics())
		cNode->m_WorldTransform = parentWM * cNode->Physics()->GetWorldSpaceTransform() * cNode->m_LocalTransform;
	else
		cNode->m_WorldTransform = parentWM * cNode->m_LocalTransform;

	for (auto child : cNode->GetChildren()) {
		UpdateWorldMatrices(child, cNode->m_WorldTransform);
	}
}

void Scene::InsertToRenderList(RenderList* list, const Frustum& frustum)
{
	InsertToRenderList(m_RootGameObject, list, frustum);
}

void Scene::InsertToRenderList(Object* node, RenderList* list, const Frustum& frustum)
{
	bool inside = frustum.InsideFrustum(node->m_WorldTransform.GetPositionVector(), node->m_BoundingRadius);

	if (inside)
	{
		//Check to see if the object is already listed or not
		if (!(list->BitMask() & node->m_FrustumCullFlags))
		{
			list->InsertObject(node);
		}
	}

	//Iterate through all child nodes
	for (auto child : node->GetChildren()) {
		InsertToRenderList(child, list, frustum);
	}
}

void Scene::UpdateNode(float dt, Object* cNode)
{
	cNode->OnUpdateObject(dt);

	for (auto child : cNode->GetChildren()) {
		UpdateNode(dt, child);
	}
}
