#version 150 core

uniform sampler2D colourTex;

uniform vec2 	singlepixel;
uniform float   numsamples;
uniform float   gammaCorrection;

in Vertex	{
	vec2 texCoord;
} IN;

out vec4 OutFrag;

vec3 supersample(sampler2D target)
{
	vec2 offtexcoord = IN.texCoord;

	//Sample every other pixel and interpolate
	vec3 col = vec3(0.0f);
	float samples_taken = 0.0f;
	for (float x = 0.0f; x < numsamples; x += 2.0f)
	{
		for (float y = 0.0f; y < numsamples; y += 2.0f)
		{
			col += texture(target, offtexcoord + singlepixel * vec2(x, y)).xyz;
			samples_taken += 1.0f;
		}
	}

	return col / samples_taken;
}

void main(void)	{

	vec3  colour		= supersample(colourTex);
	colour = pow(colour, vec3(gammaCorrection));
	
	OutFrag = vec4(colour, 1.0f);
}