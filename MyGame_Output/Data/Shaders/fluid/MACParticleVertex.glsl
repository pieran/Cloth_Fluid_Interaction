#version 150 core



in  vec4 position;
in  float colour;

out Vertex {
	vec4 pos;
	float pressure;
} OUT;

void main(void)	{
	gl_Position	  = position;
	OUT.pos		  = position;	
	OUT.pressure    = colour;
}