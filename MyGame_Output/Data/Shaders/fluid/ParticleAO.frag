#version 150 core

uniform mat4 projMatrix;
uniform float sphereRadius;
uniform sampler2D depthTex;
uniform sampler2D normalTex;
uniform vec2 screenSize;
uniform float tanHalfFOV;
uniform float sphereRadiusNrml;

in Vertex	{
	vec2 relCoords;
	vec3 eyeSpacePos;
} IN;

out vec4 gl_FragColor;

const float PI = 3.14159265359;

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
	
	
	
	
	
	//Compute Occlusion
	vec2 coordinates = gl_FragCoord.xy / screenSize;

	vec3 ss_normal = texture(normalTex, coordinates).xyz;
	float z_e = ss_normal.z;
	ss_normal.z = 1.0f - sqrt(ss_normal.x * ss_normal.x + ss_normal.y * ss_normal.y);
	
    //reconstruct position
   vec3 viewRay = vec3(
        (coordinates.x * 2.0 - 1.0) * tanHalfFOV * screenSize.x / screenSize.y,
        (coordinates.y * 2.0 - 1.0) * tanHalfFOV,
        -1.0);

    vec3 viewSpacePosition = viewRay * -z_e;

    vec3 di = IN.eyeSpacePos.xyz - normal * sphereRadius - viewSpacePosition;
    float l = length(di);

    float nl = dot(ss_normal, di / l);
    float h = l / sphereRadius;
    float h2 = h * h;
    float k2 = 1.0 - h2 * nl * nl;

    float result = max(0.0, nl) / h2;

    if (k2 > 0.0 && l > sphereRadius ) {
        result = nl * acos(-nl * sqrt((h2 - 1.0) / (1.0 - nl * nl))) - sqrt(k2 * (h2 - 1.0));
        result = result / h2 + atan(sqrt(k2 / (h2 - 1.0)));
        result /= PI;

        //result = pow( clamp(0.5*(nl*h+1.0)/h2,0.0,1.0), 1.5 ); //cheap approximation
    }
	
	
	
	
	
	
	
	
	
	gl_FragColor = vec4(vec3(result), 1.0f);
}