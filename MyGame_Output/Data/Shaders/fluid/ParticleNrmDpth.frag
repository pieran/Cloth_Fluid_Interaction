#version 150 core

uniform mat4 projMatrix;
uniform float sphereRadius;

in Vertex	{
	vec2 relCoords;
	vec3 eyeSpacePos;
} IN;

out vec4 gl_FragColor;


void main(void)	{
	vec3 normal;
	normal.xy = IN.relCoords;

	float r2 = dot(normal.xy, normal.xy);
	if (r2 > 1.0f) discard; //kill pixels outside circle
	
	normal.z = sqrt(1.0f - r2);
	
	//calculate depth
	vec4 pixelPos = vec4(IN.eyeSpacePos.xyz + normal * sphereRadius, 1.0f);
	vec4 clipSpacePos = projMatrix * pixelPos;
	
	float far=gl_DepthRange.far; float near=gl_DepthRange.near;
	float ndc_depth = clipSpacePos.z / clipSpacePos.w;
	float depth = (((far-near) * ndc_depth) + near + far) / 2.0;
	gl_FragDepth = depth;
	
	normal = normalize(normal) * 0.5f + 0.5f;

	gl_FragColor = vec4(normal.x, normal.y, pixelPos.z, 1.0f);
}