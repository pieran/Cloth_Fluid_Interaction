#version 150 core

uniform sampler2D texDepth;
uniform sampler2D texAbsorb;

uniform mat4 projMatrix;
uniform mat4 invProjMatrix;

uniform vec2 screen_size;
uniform float blurScale = 5.1518148125f;
uniform float blurDepthFalloff = 5021.0f;//500.0f;

//uniform float filterRadius = 10.0f;

in Vertex	{
	vec2 texCoords;
} IN;

out vec4 gl_FragColor;

const float f = 1000.0;
const float n = 0.1;

float get_linear_depth(vec2 coords)
{
	return (2 * n) / (f + n - texture(texDepth, coords).x * (f - n));
}

void main(void)	{
	float base_depth = texture(texDepth, IN.texCoords).x;
	if (base_depth > 0.99f)
	{
		discard;
		return;
	}
	float depth = (2 * n) / (f + n - base_depth * (f - n));
	//depth = (depth- n) / f;
	float absorb = texture(texAbsorb, IN.texCoords).x;

	vec4 clip_space = vec4(IN.texCoords.x, IN.texCoords.y, base_depth, 1.0f) * 2.0f - 1.0f;
	vec4 eye_space = invProjMatrix * clip_space;
	//eye_space.xyz /= eye_space.w;
	eye_space.xy += vec2(0.7, 0.7);
	vec4 adj_clip_space = projMatrix * eye_space;
	adj_clip_space.xyz /= adj_clip_space.w;
	
	vec2 tsDistance = (adj_clip_space.xy * 0.5f - 0.5f) - IN.texCoords;
	float filterRadius = clamp(1.0f / sqrt(dot(tsDistance, tsDistance)) * 3.0f, 1.0f, 30.0f);
	
	filterRadius = 31.0f - clamp(depth * 1500.0f, 1.0f, 30.0f);
	filterRadius = 5.0f;//5.0f;
	
	float dsum = 0.0f;
	float dwsum = 0.0f;
	float asum = 0.0f;
	float awsum = 0.0f;
	
	vec2 pix = vec2(1.0f) / screen_size;
	
	for(float x=-filterRadius; x<=filterRadius; x+=1.0) {
		for(float y=-filterRadius; y<=filterRadius; y+=1.0) {
			float dsample = get_linear_depth(IN.texCoords + vec2(pix.x * x, pix.y * y));
			float asample = texture(texAbsorb, IN.texCoords + vec2(pix.x * x, pix.y * y)).x;
		
		float dist = sqrt(x *x + y * y);
			// spatial domain
			float r = dist * blurScale;
			float w = exp(-r*r);
			
			// range domain
			float r2 = (dsample - depth) * blurDepthFalloff;
			float g = exp(-r2*r2);
			
			dsum += dsample * w * g;
			dwsum += w * g;
			

			asum += asample * w * g;
			awsum += w * g;
		}
	}
	
	if (dwsum > 0.0) {
		dsum /= dwsum;
	}
	
	if (awsum > 0.0) {
		asum /= awsum;
	}	
	
	gl_FragDepth = -(((2 * n) / dsum) - f - n) / (f - n);
	gl_FragColor = vec4(1);//vec3(asum), 1.0f);
}