/******************************************************************************
Class: Constraint
Implements: 
Author: Pieran Marris <p.marris@newcastle.ac.uk>
Description:
A generic template class to represent a linear constraint. 

A rigid body has 6 degrees of freedom: 3 positional and 3 rotational. A 
constraint in this sense is anything which acts to constrain the movement of that
rigid body.

		(\_/)
		( '_')
	 /""""""""""""\=========     -----D
	/"""""""""""""""""""""""\
....\_@____@____@____@____@_/

*//////////////////////////////////////////////////////////////////////////////
#pragma once
#include "PhysicsObject.h"
#include <nclgl\Vector3.h>

class Constraint
{
public:
	Constraint() {}

	Constraint(PhysicsObject* objA, PhysicsObject* objB,
		Vector3 j1, Vector3 j2, Vector3 j3, Vector3 j4, float b) {}


	virtual void ApplyImpulse() = 0;	//Apply Delta Update
	

	//Optional overridable functions, incase the constraint values change each frame (e.g distance constraint changes direction after each position update)
	virtual void PreSolverStep(float dt) {}
	virtual void DebugDraw() const {}
};