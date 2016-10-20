#include "Object.h"
#include "PhysicsEngine.h"

Object::Object(const std::string& name)
	: m_Scene(NULL)
	, m_Parent(NULL)
	, m_Name(name)
	, m_Colour(1.0f, 1.0f, 1.0f, 1.0f)
	, m_BoundingRadius(1.0f)
	, m_FrustumCullFlags(NULL)
	, m_PhysicsObject(NULL)
{
	m_LocalTransform.ToIdentity();
	m_WorldTransform.ToIdentity();
}

Object::~Object()
{
	if (m_PhysicsObject != NULL)
	{
		PhysicsEngine::Instance()->RemovePhysicsObject(m_PhysicsObject);
		delete m_PhysicsObject;
		m_PhysicsObject = NULL;
	}
}


void Object::CreatePhysicsNode()
{
	if (m_PhysicsObject == NULL)
	{
		m_PhysicsObject = new PhysicsObject();
		m_PhysicsObject->m_Parent = this;
		PhysicsEngine::Instance()->AddPhysicsObject(m_PhysicsObject);
	}
}

Object*	Object::FindGameObject(const std::string& name)
{
	//Has this object got the same name?
	if (GetName().compare(name) == 0)
	{
		return this;
	}

	//Recursively search ALL child objects and return the first one matching the given name
	for (auto child : m_Children)
	{
		//Has the object in question got the same name?
		Object* cObj = child->FindGameObject(name);
		if (cObj != NULL)
		{
			return cObj;
		}
	}

	//Object not found with the given name
	return NULL;
}

void Object::AddChildObject(Object* child)
{
	m_Children.push_back(child);
	child->m_Parent = this;
	child->m_Scene = this->m_Scene;
}
