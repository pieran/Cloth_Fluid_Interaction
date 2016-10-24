#version 150 core

uniform mat4 mdlMatrix;
in  vec3 position;

out Vertex {
	vec3 pos;	
} OUT;




void main(void)	{
	vec4 worldPos = mdlMatrix * vec4(position, 1.0f);
	OUT.pos		  = worldPos.xyz;	
	gl_Position = worldPos;
}