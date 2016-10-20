#ifndef MAT33_H
#define MAT33_H

#include "Vector3.h"

class Matrix4;

class Matrix3
{
public:

	static const Matrix3 Identity;
	static const Matrix3 ZeroMatrix;

	//ctor
	Matrix3();

	Matrix3(float elements[9]);

	Matrix3(const Vector3& c1, const Vector3& c2, const Vector3& c3);

	Matrix3(float a1, float a2, float a3,
			float b1, float b2, float b3,
			float c1, float c2, float c3);

	Matrix3(const Matrix4& mat44);

	~Matrix3(void);




	//Default States
	void	ToZero();
	void	ToIdentity();



	//Default Accessors
	inline float   operator[](int index) const        { return mat_array[index]; }
	inline float&  operator[](int index)              { return mat_array[index]; }
	inline float   operator()(int row, int col) const { return mat_array[row + col * 3]; }
	inline float&  operator()(int row, int col)       { return mat_array[row + col * 3]; }

	inline const Vector3&	GetCol(int idx) const				{ return *((Vector3*)&mat_array[idx * 3]); }
	inline void				SetCol(int idx, const Vector3& row)	{ memcpy(&mat_array[idx * 3], &row.x, 3 * sizeof(float)); }

	inline Vector3			GetRow(int idx)	const				{ return Vector3(mat_array[idx], mat_array[3 + idx], mat_array[6 + idx]); }
	inline void				SetRow(int idx, const Vector3& col)	{ mat_array[idx] = col.x; mat_array[3 + idx] = col.y; mat_array[6 + idx] = col.z; }



	//Common Matrix Properties
	inline Vector3			GetScalingVector() const			{ return Vector3(_11, _22, _33); }
	inline void				SetScalingVector(const Vector3& in)	{ _11 = in.x, _22 = in.y, _33 = in.z; }



	//Transformation Matrix
	static Matrix3 Rotation(float degrees, const Vector3 &axis);
	static Matrix3 Scale(const Vector3 &scale);



	// Standard Matrix Functionality
	static Matrix3 Inverse(const Matrix3& rhs);
	static Matrix3 Transpose(const Matrix3& rhs);
	static Matrix3 Adjugate(const Matrix3& m);

	static Matrix3 OuterProduct(const Vector3& a, const Vector3& b);



	// Additional Functionality
	float Trace() const;
	float Determinant() const;


	void DebugDrawMatrix(const Vector3& position);


	//Other representation of data.
	union
	{
		float	mat_array[9];
		struct {
			float _11, _12, _13;
			float _21, _22, _23;
			float _31, _32, _33;
		};
	};
};

Matrix3& operator+=(Matrix3& a, const Matrix3& rhs);
Matrix3& operator-=(Matrix3& a, const Matrix3& rhs);

Matrix3 operator+(const Matrix3& a, const Matrix3& rhs);
Matrix3 operator-(const Matrix3& a, const Matrix3& rhs);
Matrix3 operator*(const Matrix3& a, const Matrix3& rhs);


Matrix3& operator+=(Matrix3& a, const float b);
Matrix3& operator-=(Matrix3& a, const float b);
Matrix3& operator*=(Matrix3& a, const float b);
Matrix3& operator/=(Matrix3& a, const float b);

Matrix3 operator+(const Matrix3& a, const float b);
Matrix3 operator-(const Matrix3& a, const float b);
Matrix3 operator*(const Matrix3& a, const float b);
Matrix3 operator/(const Matrix3& a, const float b);

Vector3 operator*(const Matrix3& a, const Vector3& b);

#endif //MAT33_H