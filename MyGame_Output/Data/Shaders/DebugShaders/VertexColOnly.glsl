//This is exactly the same as Vertex.glsl, however without outputting position.
// - This seemed to be causing issues on some nvidia graphics cards

#version 150 core

uniform mat4 projViewMatrix;

in  vec4 position;
in  vec4 colour;

out Vertex {
	vec4 colour;
} OUT;

void main(void)	{
	vec4 vp = projViewMatrix * vec4(position.xyz, 1.0f);

	gl_Position	  = vp;
	
	OUT.colour    = colour;
}