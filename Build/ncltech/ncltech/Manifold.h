/******************************************************************************
Class: Manifold
Implements:
Author: Pieran Marris <p.marris@newcastle.ac.uk>
Description:
A manifold is the surface area of the collision between two objects, which 
for the purpose of this physics engine also is used to solve all the contact
constraints between the two colliding objects.

This is done by applying a distance constraint at each of the corners of the surface
area, constraining the shapes to seperate in the next frame. This is also coupled with
additional constraints of friction and also elasticity in the form of a bias term.



		(\_/)
		( '_')
	 /""""""""""""\=========     -----D
	/"""""""""""""""""""""""\
....\_@____@____@____@____@_/

*//////////////////////////////////////////////////////////////////////////////

#pragma once

#include "PhysicsObject.h"
#include <nclgl\Vector3.h>

/* A contact constraint is actually the summation of a normal distance constraint
   along with two friction constraints going along the axes perpendicular to the collision
   normal.
*/
struct Contact
{
	float   sumImpulseContact;
	Vector3 sumImpulseFriction;

	float	elatisity_term;

	Vector3 collisionNormal;
	float	collisionPenetration;

	Vector3 relPosA;			//Position relative to objectA
	Vector3 relPosB;			//Position relative to objectB
};



class Manifold
{
public:
	Manifold(PhysicsObject* nodeA, PhysicsObject* nodeB);
	~Manifold();

	//Called whenever a new collision contact between A & B are found
	void AddContact(const Vector3& globalOnA, const Vector3& globalOnB, const Vector3& normal, const float& penetration);	

	//Sequentially solves each contact constraint
	void ApplyImpulse(float solver_factor);
	void PreSolverStep(float dt);
	

	//Debug draws the manifold surface area
	void DebugDraw() const;

	//Get the physics objects
	PhysicsObject* NodeA() { return m_NodeA; }
	PhysicsObject* NodeB() { return m_NodeB; }
protected:
	void SolveContactPoint(Contact& c, float solver_factor);
	void UpdateConstraint(Contact& c);

protected:
	PhysicsObject*			m_NodeA;
	PhysicsObject*			m_NodeB;
	std::vector<Contact>	m_Contacts;
};