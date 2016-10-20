/******************************************************************************
Class: Scene
Implements: 
Author: Pieran Marris <p.marris@newcastle.ac.uk>
Description: 

The Scene class is an extrapolation of the Scene Management tutorial 
from Graphics for Games module. It contains a SceneTree of Objects which are automatically
Culled, Rendered and Updated as needed during runtime.

With the addition of the SceneManager class, multiple scenes can cohexist within the same
program meaning the same Scene could be initialied/cleaned up multiple times. The standard procedure
for a Scene lifespan follows:-
	1. Constructor()		 [Program Start]
	2. OnInitializeScene()	 [Scene Focus]
	3. OnCleanupScene()		 [Scene Lose Focus]
	4. Deconsructor()		 [Program End]

Once an object is added to the scene via AddGameObject(), the object is managed by the Scene. 
This means that it will automatically call delete on any objects you have added when the scene 
becomes innactive (lose focus). To override this you will need to override the OnCleanupScene method
and handle cleanup of Objects yourself.


		(\_/)								
		( '_')								
	 /""""""""""""\=========     -----D		
	/"""""""""""""""""""""""\			
....\_@____@____@____@____@_/

*//////////////////////////////////////////////////////////////////////////////
#pragma once

#include <nclgl/OGLRenderer.h>
#include <nclgl/Camera.h>
#include <nclgl/Shader.h>
#include <nclgl/Frustum.h>

#include "TSingleton.h"
#include "Object.h"
#include "RenderList.h"


class Scene
{
public:
	Scene(const std::string& friendly_name); //Called once at program start - all scene initialization should be done in 'OnInitialize'
	~Scene();

	virtual void OnInitializeScene()	{}								//Called when scene is being activated, and will begin being rendered/updated. - initialize objects/physics here
	virtual void OnCleanupScene()		{ DeleteAllGameObjects(); };	//Called when scene is being swapped and will no longer be rendered/updated - remove objects/physics here
	virtual void OnUpdateScene(float dt);								//This is msec * 0.001f (e.g time relative to seconds not milliseconds)

	void DeleteAllGameObjects(); //Easiest way of cleaning up the scene - unless you need to save some game objects after scene becomes innactive for some reason.

	void AddGameObject(Object* game_object);
	Object* FindGameObject(const std::string& name);

	const std::string& GetSceneName() { return m_SceneName; }

	//Sets maximum bounds of the scene - for use in shadowing
	void  SetWorldRadius(float radius)	{ m_RootGameObject->SetBoundingRadius(radius); }


	float GetWorldRadius()				{ return m_RootGameObject->GetBoundingRadius(); }

	void BuildWorldMatrices();
	void InsertToRenderList(RenderList* list, const Frustum& frustum);


protected:

	void	UpdateWorldMatrices(Object* node, const Matrix4& parentWM);
	void	InsertToRenderList(Object* node, RenderList* list, const Frustum& frustum);
	void	UpdateNode(float dt, Object* cNode);

protected:
	std::string			m_SceneName;
	Object*				m_RootGameObject;
};