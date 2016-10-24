#version 150 core

uniform mat4 projMatrix;
uniform float sphereRadius;
uniform vec3 lightDir = normalize(vec3(0.9f, -1.0f, 0.5f));
uniform sampler2D aoTex;
uniform vec2 screenSize;
uniform int useAO;

in Vertex	{
	vec2 relCoords;
	float density;
	float pressure;
	vec3 eyeSpacePos;
} IN;

out vec4 gl_FragColor;



vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

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

	
	
	vec2 coordinates = gl_FragCoord.xy / screenSize;
    float occlusion = 1.0f;
	
	if (useAO == 1)
	{
		occlusion = clamp(texture(aoTex, coordinates).r, 0, 1.0);
		occlusion = 1.0f - occlusion * 0.5;
	}
	
	vec3 color = hsv2rgb(vec3(IN.density, 0.65f, 1.0f));
	//vec3 color = hsv2rgb(vec3(IN.pressure * 20.0f + 0.5f, 1.0f, 1.0f));
		
	float diffuse = max(0.3, dot(normal, -lightDir))* occlusion;
	gl_FragColor = vec4(color * diffuse, 1.0f);
}