#version 150 core

uniform mat4 projMatrix;
uniform vec3 color = vec3(0.2f, 0.2f, 1.0f);

uniform float sphereRadius;
uniform float sphereAlpha;
uniform vec3 lightDir = normalize(vec3(0.9f, -1.0f, 0.5f));

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
	
	
	if (sphereAlpha > 0.0f)
	{
		gl_FragColor = vec4(sphereAlpha, sphereAlpha, sphereAlpha, 1.0f)
	}
	else
	{
		float diffuse = max(0.2, dot(normal, -lightDir));
		gl_FragColor = vec4(color * diffuse, 1.0f);
	}
}