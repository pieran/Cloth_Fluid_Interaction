#version 150 core

uniform mat4 modelMatrix;
uniform mat4 viewMatrix;
uniform mat4 projMatrix;
uniform mat4 textureMatrix;

uniform vec4 nodeColour;

in  vec3 position;
in  vec2 texCoord;
in  vec3 normal;

out Vertex	{
	vec4 worldPos;
	vec2 texCoord;
	vec4 colour;
	vec3 normal;
} OUT;

void main(void)	{
	vec4 wp 		= modelMatrix * vec4(position, 1.0);
	gl_Position		= projMatrix * viewMatrix * wp;
	
	OUT.worldPos 	= wp;
	OUT.texCoord	= (textureMatrix * vec4(texCoord, 0.0, 1.0)).xy;
	OUT.colour		= nodeColour;
	
	//This is a much quicker way to calculate the rotated normal value, however it only works
	//  when the model matrix has the same scaling on all axis. If this is not the case, use the other method below.
	//OUT.normal		= mat3(modelMatrix) * normal;
	
	// Use this if your objects have different scaling values for the x,y,z axis
	OUT.normal		  = transpose(inverse(mat3(modelMatrix ))) * normal;
}