#version 150 core

// "Screen Space Fluid Rendering with Curvature Flow"

// Parameters from the vertex shader
in Vertex	{
	vec2 texCoords;
} IN;

// Textures
uniform sampler2D particleTexture;

// Uniforms
uniform vec2 screenSize;
uniform mat4 projection;

// Output
out vec4 gl_FragColor;

const float zFar = 30.0;
const float zNear = 0.01;
float get_linear_depth(float non_linear_depth)
{
    float z_n = 2.0 * non_linear_depth - 1.0;
    return 2.0 * zNear * zFar / (zFar + zNear - z_n * (zFar - zNear));
}

float myDepth;
float getDepth(vec2 coords)
{
	float depth = texture(particleTexture, coords).x;
	if (abs(get_linear_depth(myDepth) - get_linear_depth(depth)) > 0.11f)
	{
		depth = myDepth;
	}
	return depth;
}

// Mean curvature. From "Screen Space Fluid Rendering with Curvature Flow"
vec3 meanCurvature(vec2 pos) {
	// Width of one pixel
	vec2 dx = vec2(1.0f / screenSize.x, 0.0f);
	vec2 dy = vec2(0.0f, 1.0f / screenSize.y);

	// Central z value
	float zc =  getDepth(pos);

	// Take finite differences
	// Central differences give better results than one-sided here.
	// TODO better boundary conditions, possibly.
	// Remark: This is not easy, get to choose between bad oblique view smoothing
	// or merging of unrelated particles
	float zdxp = getDepth(pos + dx);
	float zdxn = getDepth(pos - dx);
	float zdx = 0.5f * (zdxp - zdxn);
	zdx = (zdxp == 0.0f || zdxn == 0.0f) ? 0.0f : zdx;

	float zdyp = getDepth(pos + dy);
	float zdyn = getDepth(pos - dy);
	float zdy = 0.5f * (zdyp - zdyn);
	zdy = (zdyp == 0.0f || zdyn == 0.0f) ? 0.0f : zdy;

	// Take second order finite differences
	float zdx2 = zdxp + zdxn - 2.0f * zc;
	float zdy2 = zdyp + zdyn - 2.0f * zc;

	// Second order finite differences, alternating variables
	float zdxpyp = getDepth(pos + dx + dy);
	float zdxnyn = getDepth(pos - dx - dy);
	float zdxpyn = getDepth(pos + dx - dy);
	float zdxnyp = getDepth(pos - dx + dy);
	float zdxy = (zdxpyp + zdxnyn - zdxpyn - zdxnyp) / 4.0f;

	// Projection transform inversion terms
	float cx = 2.0f / (screenSize.x * -projection[0][0]);
	float cy = 2.0f / (screenSize.y * -projection[1][1]);

	// Normalization term
	float d = cy * cy * zdx * zdx + cx * cx * zdy * zdy + cx * cx * cy * cy * zc * zc;
	
	// Derivatives of said term
	float ddx = cy * cy * 2.0f * zdx * zdx2 + cx * cx * 2.0f * zdy * zdxy + cx * cx * cy * cy * 2.0f * zc * zdx;
	float ddy = cy * cy * 2.0f * zdx * zdxy + cx * cx * 2.0f * zdy * zdy2 + cx * cx * cy * cy * 2.0f * zc * zdy;

	// Temporary variables to calculate mean curvature
	float ex = 0.5f * zdx * ddx - zdx2 * d;
	float ey = 0.5f * zdy * ddy - zdy2 * d;

	// Finally, mean curvature
	float h = 0.5f * ((cy * ex + cx * ey) / pow(d, 1.5f));
	
	return(vec3(zdx, zdy, h));
}

void main() {
	float particleDepth = texture(particleTexture, IN.texCoords).x;
	myDepth = particleDepth;
	
	gl_FragColor = vec4(1);
	
	float outDepth = 0.0f;
	if(particleDepth != 0.0f) {
		const float dt = 0.00035f;
		const float dzt =1000.1f;
		vec3 dxyz = meanCurvature(IN.texCoords);

		// Vary contribution with absolute depth differential - trick from pySPH
		outDepth = particleDepth + dxyz.z * dt * (1.0f + (abs(dxyz.x) + abs(dxyz.y)) * dzt);
	}
	
	gl_FragDepth = outDepth;//particleDepth;//-(((2 * n) / outDepth) - f - n) / (f - n);
}