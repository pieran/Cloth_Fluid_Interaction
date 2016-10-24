#version 150 core

uniform mat4 mdlMatrix;

in  vec3 position;
in  vec3 normal; //Velocity
in  float tangent; //Density
in  float color; //Pressure

out Vertex {
	vec3 pos;	
	float density;
	vec3 vel;
	float pressure;
} OUT;




void main(void)	{
	vec4 worldPos = mdlMatrix * vec4(position, 1.0f);
	OUT.pos		  = worldPos.xyz;	
	OUT.vel 	  = (mdlMatrix * vec4(normal, 1.0f)).xyz;
	
	float vel = sqrt(dot(normal, normal)) * 0.5f;
	const float max_hue = 1.52f;
	const float min_hue = 1.6f;
	
	OUT.density	  = mix(min_hue, max_hue, clamp(vel, 0.0f, 1.0f));
	//OUT.density	  = abs(normal.x) * 0.2f;
	//OUT.density	  = tangent * 0.0002f;
	OUT.pressure  = 0.2f + color * 0.0005f;
	gl_Position = worldPos;
}