
#pragma once
#include "TSingleton.h"
#include "PhysicsObject.h"
#include "CollisionShape.h"
#include "Manifold.h"

struct CollisionData
{
	float penetration;
	Vector3 normal;
	Vector3 pointOnPlane;
};

class CollisionDetection : public TSingleton < CollisionDetection >
{
	friend class TSingleton < CollisionDetection > ;

public:

	bool CheckSphereSphereCollision(const PhysicsObject* obj1, const PhysicsObject* obj2, const CollisionShape* shape1, const CollisionShape* shape2, CollisionData* out_coldata = NULL) const;


	bool CheckCollision(const PhysicsObject* obj1, const PhysicsObject* obj2, const CollisionShape* shape1, const CollisionShape* shape2, CollisionData* out_coldata = NULL) const;
	void BuildCollisionManifold(const PhysicsObject* obj1, const PhysicsObject* obj2, const CollisionShape* shape1, const CollisionShape* shape2, const CollisionData& coldata, Manifold* out_manifold) const;


protected:
	bool CheckCollisionAxis(const Vector3& axis, const PhysicsObject* obj1, const PhysicsObject* obj2, const CollisionShape* shape1, const CollisionShape* shape2, CollisionData* out_coldata) const;

	Vector3 GetClosestPointOnEdges(const Vector3& target, const std::vector<CollisionEdge>& edges) const;
	Vector3 PlaneEdgeIntersection(const Plane& plane, const Vector3& start, const Vector3& end) const;
	void	SutherlandHodgesonClipping(const std::list<Vector3>& input_polygon, int num_clip_planes, const Plane* clip_planes, std::list<Vector3>* out_polygon, bool removePoints) const;
};