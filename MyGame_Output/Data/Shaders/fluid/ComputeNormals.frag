#version 150 core

uniform sampler2D texDepth;
uniform mat4 invProjMatrix;
uniform vec2 texelSize;

in Vertex	{
	vec2 texCoords;
} IN;

out vec4 gl_FragColor;

const float max_depth = 0.9999f;

float getDepth(vec2 texCoords)
{
	return texture(texDepth, texCoords).x;
}

vec3 getEyeSpace(vec2 texCoords)
{
	float depth = texture(texDepth, texCoords).x;
	vec4 clip_space = vec4(texCoords.x, texCoords.y, depth, 1.0f) * 2.0f - 1.0f;
	vec4 eye_space = invProjMatrix * clip_space;
	
	return eye_space.xyz / eye_space.w;
}

void main(void)	{
	vec2 texCoords = IN.texCoords;
	texCoords.y = texCoords.y;
	
	//Validate Current Depth Coordinate
	float depth = getDepth(texCoords);
	if (depth > max_depth)
	{
		discard;
		return;
	}
	
	// calculate eye-space position from depth
	vec3 posEye = getEyeSpace(texCoords);
	
	// calculate differences
	vec3 ddx = getEyeSpace(texCoords + vec2(texelSize.x, 0.0f)) - posEye;
	vec3 ddx2 = posEye - getEyeSpace(texCoords - vec2(texelSize.x, 0.0f));
	if (abs(ddx.z) > abs(ddx2.z)) {
		ddx = ddx2;
	}
	
	vec3 ddy = getEyeSpace(texCoords + vec2(0.0f, texelSize.y)) - posEye;
	vec3 ddy2 = posEye - getEyeSpace(texCoords - vec2(0.0f, texelSize.y));
	if (abs(ddy.z) < abs(ddy2.z)) {
		ddy = ddy2;
	}
	
	// calculate normal
	vec3 n = cross(ddx, ddy);
	n = normalize(n);
	
	gl_FragColor = vec4(n * 0.5f + 0.5f, 1.0f);
}