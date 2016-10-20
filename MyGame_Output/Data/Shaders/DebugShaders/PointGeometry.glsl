#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 41) out;


uniform float pix_scalar;

const int n_edges = 20;

in Vertex	{
	vec4 colour;
	vec4 pos;
} IN[];

out Vertex	{
	vec4 colour;
} OUT;

void main()  
{  
	OUT.colour = IN[0].colour;

	float radius 		= IN[0].pos.w * 2.5f;
	vec4 centrePoint 	= gl_in[0].gl_Position;

	for (int i = 0; i < n_edges; i++)
	{
		float angle = float(i) / float(n_edges) * 6.2831853;
		
		float x = cos(angle) * radius;
		float y = sin(angle) * radius;
			
		gl_Position = centrePoint;
		EmitVertex();
		
		gl_Position = centrePoint + vec4(x * pix_scalar, y, 0,0);   
		EmitVertex();	
		

	}
	
	gl_Position = centrePoint + vec4(radius * pix_scalar, 0, 0,0);   
	EmitVertex();	
}  