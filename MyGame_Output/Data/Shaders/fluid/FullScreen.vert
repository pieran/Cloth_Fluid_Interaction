#version 150 core

in vec3 position;

void main(void)	{
	gl_Position = vec4(position.xyz,0);
}