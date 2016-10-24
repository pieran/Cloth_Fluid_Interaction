#version 330 core

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

uniform float sphereRadius = 0.5f;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;

in Vertex	{
	vec3 pos;
} IN[];

out Vertex	{
	vec2 relCoords;
	vec3 eyeSpacePos;
} OUT;

void main()  
{  
	vec4 eyeCentrePoint 	= viewMatrix * vec4(IN[0].pos, 1.0f);
	vec4 csCentrePoint = projMatrix * eyeCentrePoint;
	if (csCentrePoint.z / csCentrePoint.w > -1.0f)
	{	
		vec4 a = eyeCentrePoint + vec4(vec2(sphereRadius, sphereRadius), 0.0f, 0.0f);//cam_right_radius + cam_up_radius;
		vec4 b = eyeCentrePoint + vec4(vec2(-sphereRadius, sphereRadius), 0.0f, 0.0f);//- cam_right_radius + cam_up_radius;
		vec4 c = eyeCentrePoint + vec4(vec2(sphereRadius, -sphereRadius), 0.0f, 0.0f);//+ cam_right_radius - cam_up_radius;
		vec4 d = eyeCentrePoint + vec4(vec2(-sphereRadius, -sphereRadius), 0.0f, 0.0f);//- cam_right_radius - cam_up_radius;
		
		OUT.eyeSpacePos = eyeCentrePoint.xyz;
		

		OUT.relCoords = vec2(1.0f, 1.0f);
		//OUT.eyeSpacePos = a.xyz;
		gl_Position = projMatrix * a;
		EmitVertex();	
		
		OUT.relCoords = vec2(-1.0f, 1.0f);
		//OUT.eyeSpacePos = b.xyz;
		gl_Position = projMatrix * b;
		EmitVertex();	
		
		OUT.relCoords = vec2(1.0f, -1.0f);
		//OUT.eyeSpacePos = c.xyz;
		gl_Position = projMatrix * c;
		EmitVertex();	

		OUT.relCoords = vec2(-1.0f, -1.0f);
		//OUT.eyeSpacePos = d.xyz;
		gl_Position = projMatrix * d;
		EmitVertex();	
	}
}  