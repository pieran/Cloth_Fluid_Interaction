#include "XPBD_ConstraintLayer.h"
#include "XPBD.h"
#include "CudaUtils.h"
#include "XPBD_Kernel.cuh"


void CudaTextureWrapper::allocate(
	cudaChannelFormatDesc& desc,
	uint width, uint height)
{
	gpuErrchk(cudaMallocArray(&mem, &desc, width, height, cudaArraySurfaceLoadStore));

	cudaResourceDesc sur_desc;
	memset(&sur_desc, 0, sizeof(cudaResourceDesc));
	sur_desc.resType = cudaResourceTypeArray;
	sur_desc.res.array.array = mem;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = 0;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.mipmapFilterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	gpuErrchk(cudaCreateTextureObject(&tex, &sur_desc, &texDescr, NULL));
	gpuErrchk(cudaCreateSurfaceObject(&surface, &sur_desc));
};

void CudaTextureWrapper::allocate3D(
	cudaChannelFormatDesc& desc,
	cudaExtent ce)
{
	gpuErrchk(cudaMalloc3DArray(&mem, &desc, ce, cudaArraySurfaceLoadStore));

	cudaResourceDesc sur_desc;
	memset(&sur_desc, 0, sizeof(cudaResourceDesc));
	sur_desc.resType = cudaResourceTypeArray;
	sur_desc.res.array.array = mem;

	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = 0;
	texDescr.filterMode = cudaFilterModePoint;
	texDescr.mipmapFilterMode = cudaFilterModePoint;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	gpuErrchk(cudaCreateTextureObject(&tex, &sur_desc, &texDescr, NULL));
	gpuErrchk(cudaCreateSurfaceObject(&surface, &sur_desc));
};


uint XPBD_ConstraintLayer::GetGridIdx(const uint2& grid_ref)
{
	return (grid_ref.y*m_Step) * m_Parent->m_NumX + (grid_ref.x*m_Step);
}

uint2 XPBD_ConstraintLayer::GenGridRef(uint x, uint y)
{
	uint2 o;
	o.x = x;
	o.y = y;
	return o;
}

inline Vector3 ToVec3(const Vector4& vec4)
{
	return Vector3(vec4.x, vec4.y, vec4.z);
}





XPBD_ConstraintLayer::XPBD_ConstraintLayer()
	: m_Parent(NULL)
	, m_Step(1)
	, m_cuDistanceLambdaij(NULL)
	, m_cuBendingLambdaij(NULL)
	, m_cuDistanceConstraints(NULL)
	, m_cuBendingConstraints(NULL)
	, m_cuBendingDistanceConstraints(NULL)
#if XPBD_USE_BATCHED_CONSTRAINTS == FALSE
	, m_cuConstraintOutput(NULL)
	, m_cuConstraintLookups(NULL)	//Particle -> OutputLookups
	, m_cuConstraintOutputLookups(NULL) //Particle -> Constraint Output List
#endif
{

}

XPBD_ConstraintLayer::~XPBD_ConstraintLayer()
{
	m_Step = 0;
	m_Parent = NULL;
	Release();
}

void XPBD_ConstraintLayer::Initialize(XPBD* parent, int step)
{
	m_Parent = parent;
	m_Step = step;

	m_NumX = (m_Parent->m_NumX) / m_Step + 1;
	m_NumY = (m_Parent->m_NumY) / m_Step + 1;
	if (m_Step == 1)
	{
		m_NumX--;
		m_NumY--;
	}



	k_weft = pow(m_Parent->m_MaterialStretchWeft, m_Step);
	k_warp = pow(m_Parent->m_MaterialStretchWarp, m_Step);
	k_shear = pow(m_Parent->m_MaterialShear, m_Step);
	k_bend = pow(m_Parent->m_MaterialBend, m_Step);
}

void XPBD_ConstraintLayer::Release()
{
	m_DistanceConstraints.clear();
	m_BendingConstraints.clear();
	m_BendingDistanceConstraints.clear();

	if (m_cuDistanceConstraints != NULL)
	{
		gpuErrchk(cudaFree(m_cuDistanceConstraints));
		gpuErrchk(cudaFree(m_cuDistanceLambdaij));
		gpuErrchk(cudaFree(m_cuBendingConstraints));
		gpuErrchk(cudaFree(m_cuBendingLambdaij));
		gpuErrchk(cudaFree(m_cuBendingDistanceConstraints));
		m_cuBendingConstraints = NULL;
	}

	if (m_cuMipMapParticles.mem != NULL)
	{
		gpuErrchk(cudaFreeArray(m_cuMipMapParticles.mem));
		gpuErrchk(cudaFreeArray(m_cuMipMapParticlesOld.mem));
		m_cuMipMapParticles.mem = NULL;
	}

#if XPBD_USE_BATCHED_CONSTRAINTS == FALSE
	if (m_cuConstraintOutput)
	{
		gpuErrChk(cudaFree(m_cuConstraintOutput));
		gpuErrChk(cudaFree(m_cuConstraintLookups));
		gpuErrChk(cudaFree(m_cuConstraintOutputLookups));
		m_cuConstraintOutput = NULL;
	}
#endif
}


void XPBD_ConstraintLayer::GenerateData(Vector4* initial_positions)
{
	GenerateConstraints(initial_positions);
	GenerateCudaArrays(initial_positions);
}

void XPBD_ConstraintLayer::GenerateConstraints(Vector4* initial_positions)
{
#if XPBD_USE_BATCHED_CONSTRAINTS
	std::vector<std::vector<XPBDDistanceConstraint>> distance_batches(8); //4
	std::vector<std::vector<XPBDBendingConstraint>> bending_batches(6);
	std::vector<std::vector<XPBDBendDistConstraint>> benddist_batches(6);

	int l1 , l2;

	// Vertical
	for (l1 = 0; l1 < m_NumX; l1++)
	{
		for (l2 = 0; l2 < (m_NumY - 1); l2++) {
			auto c = BuildDistanceConstraint(initial_positions, GenGridRef(l1, l2), GenGridRef(l1, (l2 + 1)), m_Parent->m_MaterialStretchWarp);
			if (((l1 + l2) % 2) == 0)
			{
				distance_batches[0].push_back(c);
			}
			else
			{
				distance_batches[1].push_back(c);
			}
		}
	}

	// Horizontal	
	for (l1 = 0; l1 < m_NumY; l1++)
	{
		for (l2 = 0; l2 < (m_NumX - 1); l2++) {
			XPBDDistanceConstraint c = BuildDistanceConstraint(initial_positions, GenGridRef(l2, l1), GenGridRef(l2 + 1, l1), m_Parent->m_MaterialStretchWeft);
			if (((l1 + l2) % 2) == 0)
			{
				distance_batches[2].push_back(c);
			}
			else
			{
				distance_batches[3].push_back(c);
			}
		}
	}

	// Shearing distance constraint
	for (l1 = 0; l1 < (m_NumY - 1); l1++)
	{
		for (l2 = 0; l2 < (m_NumX - 1); l2++) {
			auto c1 = BuildDistanceConstraint(initial_positions, GenGridRef(l2, l1), GenGridRef(l2 + 1, l1 + 1), m_Parent->m_MaterialShear);
			auto c2 = BuildDistanceConstraint(initial_positions, GenGridRef(l2, l1 + 1), GenGridRef(l2 + 1, l1), m_Parent->m_MaterialShear);
			if ((l1 % 2) == 0)
			{
				distance_batches[4].push_back(c1);
				distance_batches[6].push_back(c2);
			}
			else
			{
				distance_batches[5].push_back(c1);
				distance_batches[7].push_back(c2);
			}
		}
	}

	//add vertical bending constraints
	for (int i = 0; i < m_NumX; i++) {
		for (int j = 0; j< m_NumY - 2; j++) {
			bending_batches[j % 3].push_back(BuildBendingConstraint(initial_positions, GenGridRef(i, j), GenGridRef(i, (j + 1)), GenGridRef(i, j + 2), m_Parent->m_MaterialBend));
			
			benddist_batches[j % 3].push_back(BuildBendingDistanceConstraint(initial_positions, GenGridRef(i, j), GenGridRef(i, (j + 1)), GenGridRef(i, j + 2), m_Parent->m_MaterialBend, m_Parent->m_MaterialStretchWarp));
		}
	}

	//add horizontal bending constraints
	for (int i = 0; i < m_NumX - 2; i++) {
		for (int j = 0; j < m_NumY; j++) {
			bending_batches[3 + i % 3].push_back(BuildBendingConstraint(initial_positions, GenGridRef(i, j), GenGridRef(i + 1, j), GenGridRef(i + 2, j), m_Parent->m_MaterialBend));

			benddist_batches[3 + i % 3].push_back(BuildBendingDistanceConstraint(initial_positions, GenGridRef(i, j), GenGridRef(i + 1, j), GenGridRef(i + 2, j), m_Parent->m_MaterialBend, m_Parent->m_MaterialStretchWeft));
		}
	}

	m_DistanceConstraintBatches.clear();
	m_BendingConstraintBatches.clear();
	m_BendingDistanceConstraintBatches.clear();

	m_DistanceConstraintBatches.resize(8);
	m_BendingConstraintBatches.resize(6);
	m_BendingDistanceConstraintBatches.resize(6);

	m_DistanceConstraints.clear();
	m_BendingConstraints.clear();
	m_BendingDistanceConstraints.clear();

	uint boffset = 0;
	for (int i = 0; i < 8; ++i)
	{
		m_DistanceConstraintBatches[i].x = m_DistanceConstraints.size();

		for (int j = 0; j < distance_batches[i].size(); ++j)
		{
			m_DistanceConstraints.push_back(distance_batches[i][j]);
		}

		m_DistanceConstraintBatches[i].y = m_DistanceConstraints.size();
	}

	for (int i = 0; i < 6; ++i)
	{
		m_BendingConstraintBatches[i].x = m_BendingConstraints.size();
		m_BendingDistanceConstraintBatches[i].x = m_BendingDistanceConstraints.size();

		for (int j = 0; j < bending_batches[i].size(); ++j)
		{
			m_BendingConstraints.push_back(bending_batches[i][j]);
		}

		for (int j = 0; j < benddist_batches[i].size(); ++j)
		{
			m_BendingDistanceConstraints.push_back(benddist_batches[i][j]);
		}

		m_BendingConstraintBatches[i].y = m_BendingConstraints.size();
		m_BendingDistanceConstraintBatches[i].y = m_BendingDistanceConstraints.size();
	}
#else
	// Horizontal
	int l1, l2;
	for (l1 = 0; l1 < m_NumY; l1++)	// v
	for (l2 = 0; l2 < (m_NumX - 1); l2++) {
		m_DistanceConstraints.push_back(BuildDistanceConstraint(initial_positions, u2(l2, l1), u2(l2 + 1, l1), m_Parent->m_MaterialStretchWeft));
	}

	// Vertical
	for (l1 = 0; l1 < (m_NumX); l1++)
	for (l2 = 0; l2 < (m_NumY - 1); l2++) {
		m_DistanceConstraints.push_back(BuildDistanceConstraint(initial_positions, u2(l1, l2), u2(l1, (l2 + 1)), m_Parent->m_MaterialStretchWarp));
	}

	// Shearing distance constraint
	for (l1 = 0; l1 < (m_NumY - 1); l1++)
	for (l2 = 0; l2 < (m_NumX - 1); l2++) {
		m_DistanceConstraints.push_back(BuildDistanceConstraint(initial_positions, u2(l2, l1), u2(l2 + 1, l1 + 1), m_Parent->m_MaterialShear));
		m_DistanceConstraints.push_back(BuildDistanceConstraint(initial_positions, u2(l2, l1 + 1), u2(l2 + 1, l1), m_Parent->m_MaterialShear));
	}

	//add vertical bending constraints
	for (int i = 0; i < m_NumX; i++) {
		for (int j = 0; j<m_NumY - 2; j++) {
			m_BendingConstraints.push_back(BuildBendingConstraint(initial_positions, u2(i, j), u2(i, (j + 1)), u2(i, j + 2), m_Parent->m_MaterialBend));
		}
	}
	//add horizontal bending constraints
	for (int i = 0; i < m_NumX - 2; i++) {
		for (int j = 0; j < m_NumY; j++) {
			m_BendingConstraints.push_back(BuildBendingConstraint(initial_positions, u2(i, j), u2(i + 1, j), u2(i + 2, j), m_Parent->m_MaterialBend));
		}
	}

	for (size_t i = 0; i < m_DistanceConstraints.size(); ++i)
	{
		m_DistanceConstraints[i].out_offset = i * 2;
	}
	size_t offset = m_DistanceConstraints.size() * 2;
	for (size_t i = 0; i < m_BendingConstraints.size(); ++i)
	{
		m_BendingConstraints[i].out_offset = offset + i * 3;
	}
#endif
}

void XPBD_ConstraintLayer::GenerateCudaArrays(Vector4* initial_positions)
{
	//Create Constraint Arrays (Distance/Bending)
	gpuErrchk(cudaMalloc((void**)&m_cuDistanceConstraints, m_DistanceConstraints.size() * sizeof(XPBDDistanceConstraint)));
	gpuErrchk(cudaMalloc((void**)&m_cuBendingConstraints, m_BendingConstraints.size() * sizeof(XPBDBendingConstraint)));
	gpuErrchk(cudaMalloc((void**)&m_cuBendingDistanceConstraints, m_BendingDistanceConstraints.size() * sizeof(XPBDBendDistConstraint)));

	if (m_DistanceConstraints.size() > 0) gpuErrchk(cudaMemcpy(m_cuDistanceConstraints, &m_DistanceConstraints[0], m_DistanceConstraints.size() * sizeof(XPBDDistanceConstraint), cudaMemcpyHostToDevice));
	if (m_BendingConstraints.size() > 0) gpuErrchk(cudaMemcpy(m_cuBendingConstraints, &m_BendingConstraints[0], m_BendingConstraints.size() * sizeof(XPBDBendingConstraint), cudaMemcpyHostToDevice));
	if (m_BendingDistanceConstraints.size() > 0) gpuErrchk(cudaMemcpy(m_cuBendingDistanceConstraints, &m_BendingDistanceConstraints[0], m_BendingDistanceConstraints.size() * sizeof(XPBDBendDistConstraint), cudaMemcpyHostToDevice));


	gpuErrchk(cudaMalloc((void**)&m_cuDistanceLambdaij, m_DistanceConstraints.size() * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&m_cuBendingLambdaij, m_BendingConstraints.size() * sizeof(float)));

	gpuErrchk(cudaMemset(m_cuDistanceLambdaij, 0, m_DistanceConstraints.size() * sizeof(float)));
	gpuErrchk(cudaMemset(m_cuBendingLambdaij, 0, m_BendingConstraints.size() * sizeof(float)));


	cudaChannelFormatDesc channelDesc4F = cudaCreateChannelDesc<float4>();


	m_cuMipMapParticles.allocate(channelDesc4F, m_NumX, m_NumY);
	m_cuMipMapParticlesOld.allocate(channelDesc4F, m_NumX, m_NumY);

#if XPBD_USE_BATCHED_CONSTRAINTS == FALSE
	//Create lock free lookups and constraint output locations
	uint num_constraint_outputs = m_DistanceConstraints.size() * 2 + m_BendingConstraints.size() * 3;

	auto get_idx = [&](uint2 coord) {
		return coord.y * m_NumX + coord.x;
	};

	std::vector<std::vector<uint>> lookups(m_NumParticles);
	for (size_t i = 0; i < m_DistanceConstraints.size(); ++i)
	{
		lookups[get_idx(m_DistanceConstraints[i].p1)].push_back(i * 2);
		lookups[get_idx(m_DistanceConstraints[i].p2)].push_back(i * 2 + 1);
	}
	size_t offset = m_DistanceConstraints.size() * 2;
	for (size_t i = 0; i < m_BendingConstraints.size(); ++i)
	{
		lookups[get_idx(m_BendingConstraints[i].p1)].push_back(offset + i * 3);
		lookups[get_idx(m_BendingConstraints[i].p2)].push_back(offset + i * 3 + 1);
		lookups[get_idx(m_BendingConstraints[i].p3)].push_back(offset + i * 3 + 2);
	}

	uint2* initial_constraint_lookups = new uint2[m_NumParticles];
	uint* initial_constraint_output_lookups = new uint[num_constraint_outputs];
	offset = 0;
	for (size_t i = 0; i < lookups.size(); ++i)
	{
		initial_constraint_lookups[i].x = offset;
		for (size_t j = 0; j < lookups[i].size(); ++j)
		{
			initial_constraint_output_lookups[offset++] = lookups[i][j];
		}
		initial_constraint_lookups[i].y = offset;
	}

	printf("Constraint Output Size: %d [%5.2fMB]", num_constraint_outputs * sizeof(float3), float(num_constraint_outputs * sizeof(float3)) / 1000000.0f);

	gpuErrchk(cudaMalloc((void**)&m_cuConstraintOutput, num_constraint_outputs * sizeof(float3)));
	gpuErrchk(cudaMalloc((void**)&m_cuConstraintLookups, m_NumParticles * sizeof(uint2)));
	gpuErrchk(cudaMalloc((void**)&m_cuConstraintOutputLookups, num_constraint_outputs * sizeof(uint)));

	gpuErrchk(cudaMemcpy(m_cuConstraintLookups, initial_constraint_lookups, m_NumParticles * sizeof(uint2), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(m_cuConstraintOutputLookups, initial_constraint_output_lookups, num_constraint_outputs * sizeof(uint), cudaMemcpyHostToDevice));

	delete[] initial_constraint_lookups;
	delete[] initial_constraint_output_lookups;
	lookups.clear();
#endif
}



XPBDDistanceConstraint XPBD_ConstraintLayer::BuildDistanceConstraint(const Vector4* positions, uint2 idxA, uint2 idxB, float k)
{
	XPBDDistanceConstraint out_constraint;

	out_constraint.p1 = idxA;
	out_constraint.p2 = idxB;
	//out_constraint.k = pow(k, m_Step);

	Vector3 deltaP = ToVec3(positions[GetGridIdx(idxA)]) - ToVec3(positions[GetGridIdx(idxB)]);
	out_constraint.rest_length = deltaP.Length();

	return out_constraint;
}

XPBDBendingConstraint XPBD_ConstraintLayer::BuildBendingConstraint(const Vector4* positions, uint2 idxA, uint2 idxB, uint2 idxC, float k)
{
	XPBDBendingConstraint out_constraint;

	out_constraint.p1 = idxA;
	out_constraint.p2 = idxB;
	out_constraint.p3 = idxC;
	//out_constraint.k = pow(k, m_Step);

	Vector3 center = (ToVec3(positions[GetGridIdx(idxA)]) + ToVec3(positions[GetGridIdx(idxB)]) + ToVec3(positions[GetGridIdx(idxC)])) / 3.0f;
	out_constraint.rest_length = (ToVec3(positions[GetGridIdx(idxC)]) - center).Length();

	return out_constraint;
}

XPBDBendDistConstraint XPBD_ConstraintLayer::BuildBendingDistanceConstraint(const Vector4* positions, uint2 idxA, uint2 idxB, uint2 idxC, float k_bend, float k_stretch)
{
	XPBDBendDistConstraint out_constraint;

	out_constraint.p1 = idxA;
	//out_constraint.k_bend = pow(k_bend, m_Step);
	//out_constraint.k_dist = pow(k_stretch, m_Step);

	Vector3 center = (ToVec3(positions[GetGridIdx(idxA)]) + ToVec3(positions[GetGridIdx(idxB)]) + ToVec3(positions[GetGridIdx(idxC)])) / 3.0f;
	out_constraint.rest_length = (ToVec3(positions[GetGridIdx(idxC)]) - center).Length();

	/*Vector3 deltaP = ToVec3(positions[GetGridIdx(idxA)]) - ToVec3(positions[GetGridIdx(idxB)]);
	out_constraint.dist_rest_length1 = deltaP.Length();

	deltaP = ToVec3(positions[GetGridIdx(idxB)]) - ToVec3(positions[GetGridIdx(idxC)]);
	out_constraint.dist_rest_length2 = deltaP.Length();*/

	return out_constraint;
}


void XPBD_ConstraintLayer::ResetLambda()
{
	gpuErrchk(cudaMemset(m_cuDistanceLambdaij, 0, m_DistanceConstraints.size() * sizeof(float)));
	gpuErrchk(cudaMemset(m_cuBendingLambdaij, 0, m_BendingConstraints.size() * sizeof(float)));
}

bool XPBD_ConstraintLayer::Solve(int iterations, cudaTextureObject_t positions, cudaTextureObject_t positionstmp, cudaStream_t stream1, cudaStream_t stream2)
{

#if XPBD_USE_BATCHED_CONSTRAINTS
	cudaTextureObject_t target_tex = (m_Step == 1) ? positions : m_cuMipMapParticles.tex;

	if (m_Step > 1)
	{
		//Copy Particles to MipMap Layer
		xpbd_kernel::copytomipmap(m_NumX, m_NumY, m_Step, positions, m_cuMipMapParticles.surface);
		//xpbd_kernel::copytomipmap(m_NumX, m_NumY, m_Step, positions, m_cuMipMapParticlesOld.surface);
		gpuErrchk(cudaMemcpyArrayToArray(m_cuMipMapParticlesOld.mem, 0, 0, m_cuMipMapParticles.mem, 0, 0, m_NumX * m_NumY * sizeof(float4), cudaMemcpyDeviceToDevice));
	}

	//Solve MipMap Layer
	for (int i = 0; i < iterations; ++i)
	{
#if 0
		for (int j = 0; j < 8; ++j)
		{
			uint2 ref = m_DistanceConstraintBatches[j];
			xpbd_kernel::batched_solvedistanceconstraints(
				ref.y - ref.x,
				&m_cuDistanceLambdaij[ref.x], &m_cuDistanceConstraints[ref.x],
				target_tex, target_tex);
		}

		for (int j = 0; j < 6; ++j)
		{
			uint2 ref = m_BendingConstraintBatches[j];
			xpbd_kernel::batched_solvebendingconstraints(
				ref.y - ref.x,
				&m_cuBendingLambdaij[ref.x], &m_cuBendingConstraints[ref.x],
				target_tex, target_tex);
		}
#else
		for (int j = 4; j < 8; ++j)
		{
			uint2 ref = m_DistanceConstraintBatches[j];
			xpbd_kernel::batched_solvedistanceconstraints(
				ref.y - ref.x,
				&m_cuDistanceLambdaij[ref.x], &m_cuDistanceConstraints[ref.x],
				k_shear,
				target_tex, target_tex);
		}

		for (int j = 0; j < 6; ++j)
		{
			//Vertical: 0-3
			//Horizontal: 3-6

			uint2 ref = m_BendingDistanceConstraintBatches[j];
			xpbd_kernel::batched_solvebendingdistanceconstraints(
				j >= 3,
				ref.y - ref.x,
				&m_cuBendingDistanceConstraints[ref.x],
				k_bend, (j >= 3) ? k_weft : k_warp,
				target_tex);
		}

		for (size_t j = 0; j < m_Parent->m_SphereConstraints.size(); ++j)
		{
			xpbd_kernel::solvesphereconstraint(m_NumX, m_NumY, m_Parent->m_SphereConstraints[j], target_tex);
		}
#endif
	}



	if (m_Step > 1)
	{
		//Interpolate Results back to mipmap0
		xpbd_kernel::extrapolatemipmap(m_Parent->m_NumX, m_Parent->m_NumY, m_Step, m_cuMipMapParticles.tex, m_cuMipMapParticlesOld.tex, positions);
	}

	return false;
#else
	float weighting = i * 1.0f / float(num_iterations);
	weighting = weighting* 2.0f + 0.01f;
	xpbd_kernel::solvedistanceconstraints(m_DistanceConstraints.size(), m_cuDistanceLambdaij, m_cuDistanceConstraints, m_cuParticlePos.tex, m_cuConstraintOutput);
	xpbd_kernel::solvebendingconstraints(m_BendingConstraints.size(), m_cuBendingLambdaij, m_cuBendingConstraints, m_cuParticlePos.tex, m_cuConstraintOutput);
	xpbd_kernel::mergeoutputs(m_NumX, m_NumY, weighting, m_cuParticlePos.tex, m_cuParticlePosTmp.surface, m_cuConstraintLookups, m_cuConstraintOutputLookups, m_cuConstraintOutput);
	swap(m_cuParticlePos, m_cuParticlePosTmp);
#endif
}