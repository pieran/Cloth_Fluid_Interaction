#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform mat4 mvpMatrix;
uniform vec3 camLeft;
uniform vec3 camUp;


in Vertex	{
	vec4 pos;
	float pressure;
} IN[];

out Vertex	{
	float pressure;
	vec2 texCoords;
} OUT;

void main()  
{  
	OUT.pressure = IN[0].pressure;

	vec4 centrePoint 	= gl_in[0].gl_Position;
	vec4 a = vec4(centrePoint.xyz + camLeft + camUp, centrePoint.w);
	vec4 b = vec4(centrePoint.xyz - camLeft + camUp, centrePoint.w);
	vec4 c = vec4(centrePoint.xyz + camLeft - camUp, centrePoint.w);
	vec4 d = vec4(centrePoint.xyz - camLeft - camUp, centrePoint.w);
	
		
	gl_Position = mvpMatrix * a; 
	OUT.texCoords = vec2(1.0f, 1.0f);
	EmitVertex();	
	
	gl_Position = mvpMatrix * b; 
	OUT.texCoords = vec2(0.0f, 1.0f);
	EmitVertex();
	
	gl_Position = mvpMatrix * c; 
	OUT.texCoords = vec2(1.0f, 0.0f);
	EmitVertex();
	
	gl_Position = mvpMatrix * d; 
	OUT.texCoords = vec2(0.0f, 0.0f);
	EmitVertex();
}  