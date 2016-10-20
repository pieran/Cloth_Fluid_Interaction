/******************************************************************************
Class: PhysicsEngine
Implements:
Author: Rich Davison <richard.davison4@newcastle.ac.uk> and YOU!
Description:
A very simple physics engine class, within which to implement the
material introduced in the Game Technologies module. This is just a rough
skeleton of how the material could be integrated into the existing codebase -
it is still entirely up to you how the specifics should work. Now C++ and
graphics are out of the way, you should be starting to get a feel for class
structures, and how to communicate data between systems.

It is worth poinitng out that the PhysicsEngine is constructed and destructed
manually using static functions. Why? Well, we probably only want a single
physics system to control the entire state of our game world, so why allow
multiple systems to be made? So instead, the constructor / destructor are
hidden, and we 'get' a single instance of a physics system with a getter.
This is known as a 'singleton' design pattern, and some developers don't like
it - but for systems that it doesn't really make sense to have multiples of,
it is fine!

		(\_/)
		( '_')
	 /""""""""""""\=========     -----D
	/"""""""""""""""""""""""\
....\_@____@____@____@____@_/

*//////////////////////////////////////////////////////////////////////////////

#pragma once
#include "TSingleton.h"
#include "PhysicsObject.h"
#include "Constraint.h"
#include "Manifold.h"
#include <vector>
#include <mutex>


#define SOLVER_ITERATIONS 20


#define FALSE	0
#define TRUE	1


#define DEBUHDRAW_FLAGS_CONSTRAINT				0x1
#define DEBUHDRAW_FLAGS_MANIFOLD				0x2
#define DEBUHDRAW_FLAGS_COLLISIONVOLUMES		0x4
#define DEBUHDRAW_FLAGS_COLLISIONNORMALS		0x8


struct CollisionPair	//Forms the output of the broadphase collision detection
{
	PhysicsObject* objectA;
	PhysicsObject* objectB;
};

class PhysicsEngine : public TSingleton<PhysicsEngine>
{
	friend class TSingleton < PhysicsEngine > ;
public:
	//Reset Default Values like gravity/timestep - called when scene is switched out
	void SetDefaults();

	//Add/Remove Physics Objects
	void AddPhysicsObject(PhysicsObject* obj);
	void RemovePhysicsObject(PhysicsObject* obj);
	void RemoveAllPhysicsObjects(); //Delete all physics entities etc and reset-physics environment for new scene to be initialized

	//Add Constraints
	void AddConstraint(Constraint* c) { m_Constraints.push_back(c); }
	

	//Update Physics Engine
	void Update(float deltaTime);			//Remember DeltaTime is 'seconds' since last update not milliseconds
	
	//Debug draw all physics objects, manifolds and constraints
	void DebugRender();



	//Getters / Setters 
	bool IsPaused()						{ return m_IsPaused; }
	void SetPaused(bool paused)			{ m_IsPaused = paused; }

	uint GetDebugDrawFlags()			{ return m_DebugDrawFlags;  }
	void SetDebugDrawFlags(uint flags)  { m_DebugDrawFlags = flags; }
	
	void SetUpdateTimestep(float updateTimestep) { m_UpdateTimestep = updateTimestep; }
	float GetUpdateTimestep()			{ return m_UpdateTimestep; }

	const Vector3& GetGravity()			{ return m_Gravity; }
	void SetGravity(const Vector3& g)	{ m_Gravity = g; }

	float GetDampingFactor()			{ return m_DampingFactor; }
	void  SetDampingFactor(float d)		{ m_DampingFactor = d; }

	float GetDeltaTime()				{ return m_UpdateTimestep; }

protected:
	PhysicsEngine();
	~PhysicsEngine();

	//The actual time-independant update function
	void UpdatePhysics();

	//Handles broadphase collision detection
	void BroadPhaseCollisions();

	//Handles narrowphase collision detection
	void NarrowPhaseCollisions();
	void NarrowPhaseCollisionsBatch(size_t batch_start, size_t batch_end); //<--- The worker function for multithreading


	//Updates all physics objects position, orientation, velocity etc (default method uses symplectic euler integration)
	void UpdatePhysicsObjects();	
	void UpdatePhysicsObjectsBatch(size_t batch_start, size_t batch_end);  //<--- The worker function for multithreading
	void UpdatePhysicsObject(PhysicsObject* obj);						   //<--- The actual code to update the given physics object
	
	//Solves all engine constraints (constraints and manifolds)
	void SolveConstraints();

protected:
	bool		m_IsPaused;
	float		m_UpdateTimestep, m_UpdateAccum;
	uint		m_DebugDrawFlags;

	Vector3		m_Gravity;
	float		m_DampingFactor;


	std::vector<CollisionPair> m_BroadphaseCollisionPairs;

	std::vector<PhysicsObject*> m_PhysicsObjects;

	std::vector<Constraint*>	m_Constraints;			// Misc constraints between pairs of objects
	std::vector<Manifold*>		m_Manifolds;			// Contact constraints between pairs of objects

	std::mutex					m_ManifoldsMutex;
};