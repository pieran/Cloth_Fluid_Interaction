#version 150 core

// Parameters from the vertex shader
in Vertex	{
	vec2 texCoords;
} IN;

// Textures
uniform sampler2D particleTexture;

// Uniforms
uniform vec2 screenSize;
uniform vec2 blurDirection;

// Output
out vec4 gl_FragColor;


//https://github.com/Jam3/glsl-fast-gaussian-blur/blob/master/13.glsl
float blur13(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  float color = 0.0;
  vec2 off1 = vec2(1.411764705882353) * direction;
  vec2 off2 = vec2(3.2941176470588234) * direction;
  vec2 off3 = vec2(5.176470588235294) * direction;
  color += texture(image, uv).x * 0.1964825501511404;
  color += texture(image, uv + (off1 / resolution)).x * 0.2969069646728344;
  color += texture(image, uv - (off1 / resolution)).x * 0.2969069646728344;
  color += texture(image, uv + (off2 / resolution)).x * 0.09447039785044732;
  color += texture(image, uv - (off2 / resolution)).x * 0.09447039785044732;
  color += texture(image, uv + (off3 / resolution)).x * 0.010381362401148057;
  color += texture(image, uv - (off3 / resolution)).x * 0.010381362401148057;
  return color;
}


void main() {
	float blur_val = blur13(particleTexture, IN.texCoords, screenSize, blurDirection);
	//blur_val
	gl_FragColor = vec4(vec3(blur_val), 1.0f);
}