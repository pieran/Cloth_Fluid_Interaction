/******************************************************************************
Class: BoundingBox
Implements:
Author: Pieran Marris <p.marris@newcastle.ac.uk>
Description:
The simplest axis-aligned bounding box implementation you will ever see! 

This serves the single purpose of assisting in the creation of shadow map bounding boxes
for graphics pipeline.

Much better implementations with various different storage methods are out there and I 
highly suggest you go and find them. :)

		(\_/)
		( '_')
	/""""""""""""\=========     -----D
	/"""""""""""""""""""""""\
....\_@____@____@____@____@_/

*//////////////////////////////////////////////////////////////////////////////
#pragma once
#include <nclgl\Matrix4.h>
#include <nclgl\Vector3.h>
#include <nclgl\common.h>

class BoundingBox
{
public:
	Vector3 minPoints;
	Vector3 maxPoints;


	//Initialize minPoints to max possible value and vice versa to force the first value incorporated to always be used for both min and max points.
	BoundingBox() 
		: minPoints(FLT_MAX, FLT_MAX, FLT_MAX)
		, maxPoints(-FLT_MAX, -FLT_MAX, -FLT_MAX)
	{}

	//Expand the boundingbox to fit a given point. 
	//  If no points have been set yet, both minPoints and maxPoints will equal point.
	void ExpandToFit(const Vector3& point)
	{
		minPoints.x = min(minPoints.x, point.x);
		minPoints.y = min(minPoints.y, point.y);
		minPoints.z = min(minPoints.z, point.z);
		maxPoints.x = max(maxPoints.x, point.x);
		maxPoints.y = max(maxPoints.y, point.y);
		maxPoints.z = max(maxPoints.z, point.z);
	}

	//Transform the given AABB and returns a new AABB that encapsulates the new rotated bounding box.
	BoundingBox Transform(const Matrix4& mtx)
	{
		BoundingBox bb;
		bb.ExpandToFit(mtx * Vector3(minPoints.x, minPoints.y, minPoints.z));
		bb.ExpandToFit(mtx * Vector3(maxPoints.x, minPoints.y, minPoints.z));
		bb.ExpandToFit(mtx * Vector3(minPoints.x, maxPoints.y, minPoints.z));
		bb.ExpandToFit(mtx * Vector3(maxPoints.x, maxPoints.y, minPoints.z));

		bb.ExpandToFit(mtx * Vector3(minPoints.x, minPoints.y, maxPoints.z));
		bb.ExpandToFit(mtx * Vector3(maxPoints.x, minPoints.y, maxPoints.z));
		bb.ExpandToFit(mtx * Vector3(minPoints.x, maxPoints.y, maxPoints.z));
		bb.ExpandToFit(mtx * Vector3(maxPoints.x, maxPoints.y, maxPoints.z));
		return bb;
	}
};