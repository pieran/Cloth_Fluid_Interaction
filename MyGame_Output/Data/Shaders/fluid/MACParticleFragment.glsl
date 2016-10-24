#version 150 core

uniform sampler2D particleTex;

in Vertex	{
	float pressure;
	vec2 texCoords;
} IN;

out vec4 gl_FragColor;

void main(void)	{
	vec4 col = texture(particleTex, IN.texCoords);
	gl_FragColor.a = IN.pressure * col.a;
	gl_FragColor.rgb = (col.rgb * IN.pressure);
	
}