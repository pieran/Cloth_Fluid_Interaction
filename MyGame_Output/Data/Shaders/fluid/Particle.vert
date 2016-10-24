#version 150 core

uniform mat4 mdlMatrix;

in  vec3 position;
in  float tangent; //Density

out Vertex {
	vec3 pos;	
	float density;
} OUT;




void main(void)	{
	vec4 worldPos = mdlMatrix * vec4(position, 1.0f);
	OUT.pos		  = worldPos.xyz;
	OUT.density = tangent;	
	gl_Position = worldPos;
}