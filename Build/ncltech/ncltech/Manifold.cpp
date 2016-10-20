#include "Manifold.h"
#include <nclgl\Matrix3.h>
#include "NCLDebug.h"
#include "PhysicsEngine.h"

#define persistentThresholdSq 0.025f

typedef std::list<Contact> ContactList;
typedef ContactList::iterator ContactListItr;

Manifold::Manifold(PhysicsObject* nodeA, PhysicsObject* nodeB) : m_NodeA(nodeA), m_NodeB(nodeB)
{
}

Manifold::~Manifold()
{

}

void Manifold::ApplyImpulse(float solver_factor)
{
	float softness = (m_NodeA->GetInverseMass() + m_NodeB->GetInverseMass()) / m_Contacts.size();
	for (Contact& contact : m_Contacts)
	{
		SolveContactPoint(contact, solver_factor);
	}
}

void Manifold::SolveContactPoint(Contact& c, float solver_factor)
{
	if (m_NodeA->GetInverseMass() + m_NodeB->GetInverseMass() == 0.0f)
		return;

	Vector3 r1 = c.relPosA;
	Vector3 r2 = c.relPosB;

	Vector3 v0 = m_NodeA->GetLinearVelocity() + Vector3::Cross(m_NodeA->GetAngularVelocity(), r1);
	Vector3 v1 = m_NodeB->GetLinearVelocity() + Vector3::Cross(m_NodeB->GetAngularVelocity(), r2);

	Vector3 normal = c.collisionNormal;
	Vector3 dv = v0 - v1;

	//Collision Resolution
	{
		float constraintMass = (m_NodeA->GetInverseMass() + m_NodeB->GetInverseMass()) +
			Vector3::Dot(normal,
			Vector3::Cross(m_NodeA->GetInverseInertia()*Vector3::Cross(r1, normal), r1) +
			Vector3::Cross(m_NodeB->GetInverseInertia()*Vector3::Cross(r2, normal), r2));

		//Baumgarte Offset (Adds energy to the system to counter slight solving errors that accumulate over time - known as 'constraint drift')
		float b = 0.0f;
		{
			float distance_offset = c.collisionPenetration;
			float baumgarte_scalar = 0.3f;
			float baumgarte_slop = 0.00f;
			float penetration_slop = min(c.collisionPenetration + baumgarte_slop, 0.0f);
			b = -(baumgarte_scalar / PhysicsEngine::Instance()->GetDeltaTime()) * penetration_slop;
		}

		float jn = -(Vector3::Dot(dv, normal) + c.elatisity_term + b) / constraintMass * solver_factor;

		//As this is run multiple times per frame,
		// we need to clamp the total amount of movement to be positive
		// otherwise in some scenarios we may end up solving the constraint backwards 
		// to compensate for collisions with other objects
		float oldSumImpulseContact = c.sumImpulseContact;
		c.sumImpulseContact = min(c.sumImpulseContact + jn, 0.0f);
		float real_jn = c.sumImpulseContact - oldSumImpulseContact;


		m_NodeA->SetLinearVelocity(m_NodeA->GetLinearVelocity() + normal*(real_jn*m_NodeA->GetInverseMass()));
		m_NodeB->SetLinearVelocity(m_NodeB->GetLinearVelocity() - normal*(real_jn*m_NodeB->GetInverseMass()));

		m_NodeA->SetAngularVelocity(m_NodeA->GetAngularVelocity() + m_NodeA->GetInverseInertia()* Vector3::Cross(r1, normal * real_jn));
		m_NodeB->SetAngularVelocity(m_NodeB->GetAngularVelocity() - m_NodeB->GetInverseInertia()* Vector3::Cross(r2, normal * real_jn));
	}


	//Friction
	{
		Vector3 tangent = dv - normal * Vector3::Dot(dv, normal);
		float tangent_len = tangent.Length();

		if (tangent_len > 0.001f)
		{
			tangent = tangent * (1.0f / tangent_len);

			float tangDiv = (m_NodeA->GetInverseMass() + m_NodeB->GetInverseMass()) +
				Vector3::Dot(tangent,
				Vector3::Cross(m_NodeA->GetInverseInertia()* Vector3::Cross(r1, tangent), r1) +
				Vector3::Cross(m_NodeB->GetInverseInertia()* Vector3::Cross(r2, tangent), r2));

			float frictionCoef = (m_NodeA->GetFriction() * m_NodeB->GetFriction()) / m_Contacts.size();
			float jt = -1 * frictionCoef * Vector3::Dot(dv, tangent) / tangDiv;

			//Stop Friction from ever being more than frictionCoef * normal resolution impulse
			//
			// Similar to above for SumImpulseContact, except for friction the direction of friction solving is changing each call
			// as it is based off the objects current velocities, so must be computed as a summed radius of all friction resolution passes.
			float oldImpulseFriction = c.sumImpulseFriction.Length();
			c.sumImpulseFriction = c.sumImpulseFriction + tangent * jt;
			float diff = abs(c.sumImpulseContact) / c.sumImpulseFriction.Length() * frictionCoef;
			float real_jt = jt * min(diff, 1.0f);


			m_NodeA->SetLinearVelocity(m_NodeA->GetLinearVelocity() + tangent*(real_jt*m_NodeA->GetInverseMass()));
			m_NodeB->SetLinearVelocity(m_NodeB->GetLinearVelocity() - tangent*(real_jt*m_NodeB->GetInverseMass()));

			m_NodeA->SetAngularVelocity(m_NodeA->GetAngularVelocity() + m_NodeA->GetInverseInertia()* Vector3::Cross(r1, tangent * real_jt));
			m_NodeB->SetAngularVelocity(m_NodeB->GetAngularVelocity() - m_NodeB->GetInverseInertia()* Vector3::Cross(r2, tangent * real_jt));
		}
	}
}

void Manifold::PreSolverStep(float dt)
{
	for (Contact& contact : m_Contacts)
	{
		UpdateConstraint(contact);
	}
}

void Manifold::UpdateConstraint(Contact& contact)
{
	//Reset total impulse forces computed this physics timestep 
	contact.sumImpulseContact = 0.0f;
	contact.sumImpulseFriction = Vector3(0.0f, 0.0f, 0.0f);


	//Compute Elasticity Term
	// - Must be computed prior to solving as otherwise as the collision resolution occurs and 
	//   the velocities diverge, the elasticity_term will tend towards zero.
	{
		const float elasticity = m_NodeA->GetElasticity() * m_NodeB->GetElasticity();

		float elatisity_term = elasticity * Vector3::Dot(contact.collisionNormal,
			m_NodeA->GetLinearVelocity()
			+ Vector3::Cross(contact.relPosA, m_NodeA->GetAngularVelocity())
			- m_NodeB->GetLinearVelocity()
			- Vector3::Cross(contact.relPosB, m_NodeB->GetAngularVelocity())
			);

		//Elasticity slop here is used to make objects come to rest quicker. 
		// It works out if the elastic term is less than a given value (0.5 m/s here)
		// and if it is, then it is too small to see and ignores the elasticity calculation.
		// Most noticable when you have a stack of objects, without this they will jitter alot.
		const float elasticity_slop = 0.2f;
		if (elatisity_term < elatisity_term)
			elatisity_term = 0.0f;

		contact.elatisity_term = elatisity_term;
	}

}

void Manifold::AddContact(const Vector3& globalOnA, const Vector3& globalOnB, const Vector3& normal, const float& penetration)
{
	//Get relative offsets from each object centre of mass
	// Used to compute rotational velocity at the point of contact.
	Vector3 r1 = (globalOnA - m_NodeA->GetPosition());
	Vector3 r2 = (globalOnB - m_NodeB->GetPosition());

	//Create our new contact descriptor
	Contact contact;
	contact.relPosA = r1;
	contact.relPosB = r2;
	contact.collisionNormal = normal;
	contact.collisionPenetration = penetration;


	//Check to see if we already contain a contact point almost in that location
	const float min_allowed_dist_sq = 0.2f * 0.2f;
	bool should_add = true;
	for (auto itr = m_Contacts.begin(); itr != m_Contacts.end(); )
	{
		Vector3 ab = itr->relPosA - contact.relPosA;
		float distsq = Vector3::Dot(ab, ab);


		//Choose the contact point with the largest penetration and therefore the largest collision response
		if (distsq < min_allowed_dist_sq)
		{
			if (itr->collisionPenetration > contact.collisionPenetration)
			{
				itr = m_Contacts.erase(itr);
				continue;
			}
			else
			{
				should_add = false;
			}
			
		}
		
		itr++;
	}


	
	if (should_add)
		m_Contacts.push_back(contact);
}

void Manifold::DebugDraw() const
{
	if (m_Contacts.size() > 0)
	{
		//Loop around all contact points and draw them all as a line-fan
		Vector3 globalOnA1 = m_NodeA->GetPosition() + m_Contacts.back().relPosA;
		for (const Contact& contact : m_Contacts)
		{
			Vector3 globalOnA2 = m_NodeA->GetPosition() + contact.relPosA;
			Vector3 globalOnB = m_NodeB->GetPosition() + contact.relPosB;

			//Draw line to form area given by all contact points
			NCLDebug::DrawThickLineNDT(globalOnA1, globalOnA2, 0.02f, Vector4(0.0f, 1.0f, 0.0f, 1.0f));

			//Draw descriptors for indivdual contact point
			NCLDebug::DrawPointNDT(globalOnA2, 0.05f, Vector4(0.0f, 0.5f, 0.0f, 1.0f));
			NCLDebug::DrawThickLineNDT(globalOnB, globalOnA2, 0.01f, Vector4(1.0f, 0.0f, 1.0f, 1.0f));

			globalOnA1 = globalOnA2;
		}
	}
}