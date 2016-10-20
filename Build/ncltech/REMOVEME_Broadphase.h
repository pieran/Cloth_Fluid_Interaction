
#pragma once
#include "ncltech\TSingleton.h"
#include <nclgl\Vector3.h>
#include "ncltech\PhysicsObject.h"
#include "ncltech\BoundingBox.h"
#include "ncltech\SphereCollisionShape.h"
#include "ncltech\CuboidCollisionShape.h"
#include "ncltech\Object.h"
#include "ncltech\NCLDebug.h"
#include "ncltech\PhysicsEngine.h"


#define MAX_OCTREE_LEVEL 8

class REMOVEME_Broadphase : public TSingleton<REMOVEME_Broadphase>
{
	friend class TSingleton<REMOVEME_Broadphase>;

public:

	void ClearAll()
	{
		DeleteRecursive(&m_Root);
		m_Root = new OctNode();
	}

	void UpdateObject(PhysicsObject* obj)
	{
		if (obj->broadphase_ptr != NULL)
		{
			OctNode* node = (OctNode*)obj->broadphase_ptr;

			for (auto itr = node->objects.begin(), end = node->objects.end(); itr != end; ++itr)
			{
				if (itr->obj == obj)
				{
					node->objects.erase(itr);
					/*if (node->objects.size() == 0 && node->parent != NULL)
					{
						node->parent->children[node->idx] = NULL;
						delete node;
					}
					*/
					AddObject(obj);
					return;
				}
			}
		}

		AddObject(obj);
	}

	void AddObject(PhysicsObject* obj)
	{
		CollisionShape* shape = obj->GetCollisionShape();
		if (shape != NULL)
		{
			float radius = 0.0f;
			Object* scene_obj = obj->GetGameObject();
			if (scene_obj == NULL)
			{
				if (obj->GetCollisionShape() != NULL)
				{
					SphereCollisionShape* shape = dynamic_cast<SphereCollisionShape*>(obj->GetCollisionShape());
					if (shape != NULL)
					{
						radius = shape->GetRadius();
					}
					else
					{
						CuboidCollisionShape* shape = (CuboidCollisionShape*)(obj->GetCollisionShape());
						radius = shape->GetHalfDims().Length();
					}
				}
				else
				{
					//ERROR
					return;
				}
			}
			else
				radius = scene_obj->GetBoundingRadius();


			OctNode_Object oobj;
			oobj.obj = obj;
			oobj.radius = radius;

			BoundingBox aabb;
			aabb.minPoints = -m_WorldSize;
			aabb.maxPoints = m_WorldSize;

			AddObjectRecursive(1, aabb, m_Root, oobj);
		}
	}

	void BuildCollisionPairs(std::vector<CollisionPair>& pairs)
	{
		BuildCollisionPairsRecursive(m_Root, pairs);
	}

protected:
	struct OctNode_Object
	{
		PhysicsObject* obj;
		float radius;
	};

	struct OctNode
	{
		OctNode()
		: parent(NULL)
		, idx(0)
		{
			memset(&children[0], 0, sizeof(OctNode*)* 8);
		}

		OctNode* parent;
		char idx;
		OctNode* children[8];
		std::vector<OctNode_Object> objects;
	};


	REMOVEME_Broadphase()
	{
		m_Root = new OctNode();

		m_WorldSize = Vector3(80.0f, 80.0f, 80.0f);
	}

	~REMOVEME_Broadphase()
	{
		DeleteRecursive(&m_Root);
	}

	void DeleteRecursive(OctNode** node)
	{
		for (int i = 0; i < 8; ++i)
		if ((*node)->children[i] != NULL)
			DeleteRecursive(&(*node)->children[i]);

		delete *node;
		*node = NULL;
	}



	void AddObjectRecursive(int level, const BoundingBox& aabb, OctNode* node, OctNode_Object& obj)
	{
		Vector3 pos = obj.obj->GetPosition();
		float& r = obj.radius;

		const Vector3& min = aabb.minPoints;
		const Vector3& max = aabb.maxPoints;
		Vector3 mid = (aabb.maxPoints + aabb.minPoints) * 0.5f;
		auto overlap_axis = [&](const float& pos, float radius, float mid)
		{
			return (pos - radius < mid && pos + radius > mid);
		};

		if (level == MAX_OCTREE_LEVEL
			|| overlap_axis(pos.x, r, mid.x)
			|| overlap_axis(pos.y, r, mid.y)
			|| overlap_axis(pos.z, r, mid.z))
		{
			obj.obj->broadphase_ptr = node;
			node->objects.push_back(obj);
		}
		else
		{
			int child_index = 0;
			BoundingBox new_aabb;
			new_aabb.minPoints = min;
			new_aabb.maxPoints = mid;

			if (pos.x > mid.x)
			{
				child_index |= 1;
				new_aabb.minPoints.x = mid.x;
				new_aabb.maxPoints.x = max.x;
			}

			if (pos.y > mid.y)
			{
				child_index |= 2;
				new_aabb.minPoints.y = mid.y;
				new_aabb.maxPoints.y = max.y;
			}

			if (pos.z > mid.z)
			{
				child_index |= 4;
				new_aabb.minPoints.z = mid.z;
				new_aabb.maxPoints.z = max.z;
			}


			if (node->children[child_index] == NULL)
			{
				node->children[child_index] = new OctNode();
				node->children[child_index]->idx = child_index;
				node->children[child_index]->parent = node;
			}


			AddObjectRecursive(level + 1, new_aabb, node->children[child_index], obj);
		}
	}


	void HandleCheck(std::vector<CollisionPair>& pairs, const OctNode_Object *objA, const OctNode_Object *objB)
	{
		float dist_sq = (objA->obj->GetPosition() - objB->obj->GetPosition()).LengthSquared();
		float radii = (objA->radius + objB->radius);
		radii = radii * radii;
		if (dist_sq < radii)
		{
			CollisionPair pair;
			pair.objectA = objA->obj;
			pair.objectB = objB->obj;
			pairs.push_back(pair);
		}
	}

	void CollideChildren(std::vector<CollisionPair>& pairs, const std::vector<OctNode_Object>& objects, OctNode* node)
	{
		const OctNode_Object *objA, *objB;
		size_t size = objects.size();
		size_t child_size = node->objects.size();
		for (size_t i = 0; i < size; ++i)
		{
			objA = &objects[i];
			for (size_t j = 0; j < child_size; ++j)
			{
				objB = &node->objects[j];

				if (objA->obj->awake || objB->obj->awake)
					HandleCheck(pairs, objA, objB);
			}
		}

		for (size_t i = 0; i < 8; ++i)
		{
			if (node->children[i] != NULL)
				CollideChildren(pairs, objects, node->children[i]);
		}
	}

	void BuildCollisionPairsRecursive(OctNode* node, std::vector<CollisionPair>& pairs)
	{
		OctNode_Object *objA, *objB;
		size_t size = node->objects.size();


		//Build Collision Pairs between children	
		if (size > 0)
		{
			for (size_t i = 0; i < size - 1; ++i)
			{
				objA = &node->objects[i];
				for (size_t j = i + 1; j < size; ++j)
				{
					objB = &node->objects[j];

					if (objA->obj->awake || objB->obj->awake)
						HandleCheck(pairs, objA, objB);
				}
			}
		}

		for (size_t i = 0; i < 8; ++i)
		{
			if (node->children[i] != NULL)
			{
				CollideChildren(pairs, node->objects, node->children[i]);
				BuildCollisionPairsRecursive(node->children[i], pairs);
			}
				
		}
	}
protected:
	Vector3 m_WorldSize;
	OctNode* m_Root;
};