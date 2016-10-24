#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

out Vertex	{
	vec2 texCoords;
} OUT;

void main()  
{  
	vec4 a = vec4(1.0f, 1.0f, 0.0f, 1.0f);
	vec4 b = vec4(-1.0f, 1.0f, 0.0f, 1.0f);
	vec4 c = vec4(1.0f, -1.0f, 0.0f, 1.0f);
	vec4 d = vec4(-1.0f, -1.0f, 0.0f, 1.0f);
	


	OUT.texCoords = vec2(1.0f, 1.0f);
	gl_Position = a;
	EmitVertex();	
	
	OUT.texCoords = vec2(0.0f, 1.0f);
	gl_Position = b;
	EmitVertex();	
	
	OUT.texCoords = vec2(1.0f, 0.0f);
	gl_Position = c;
	EmitVertex();	

	OUT.texCoords = vec2(0.0f, 0.0f);
	gl_Position = d;
	EmitVertex();	
}  