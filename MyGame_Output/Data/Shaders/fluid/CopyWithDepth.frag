#version 150 core

uniform sampler2D texColor;
uniform sampler2D texDepth;

in Vertex	{
	vec2 texCoords;
} IN;

out vec4 gl_FragColor;

void main(void)	{
	float ndc_depth = texture(texDepth, IN.texCoords).x;
	if (ndc_depth > 0.99999f)
	{
		discard;
		return;
	}
	
	gl_FragDepth = ndc_depth;
	gl_FragColor = texture(texColor, IN.texCoords);
}