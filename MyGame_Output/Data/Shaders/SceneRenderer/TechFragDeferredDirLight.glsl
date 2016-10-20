#version 150 core

uniform sampler2D depthTex;
uniform sampler2D colourTex;
uniform sampler2D normalTex;

uniform mat4  clipToWorldTransform;

uniform vec3  invLightDir;			//Directional Light
uniform vec3  cameraPos;
uniform float specularIntensity;

uniform int shadowNum;
uniform sampler2DShadow shadowTex[16];
uniform mat4 shadowTransform[16];
uniform vec2 shadowSinglePixel = vec2(1.0f / 2048.0f);

in Vertex	{
	vec2 texCoord;
} IN;

out vec4 OutFrag;

float shadowTest(vec3 tsShadow, float tsShadowW, sampler2DShadow shadowTex, mat4 shadowTransform)
{	
	/*
	PCF filtering
	  - Takes a 4x4 sample around each pixel and averages the result, bluring the edges of shadows
	  - Requires 16 linearly interpolated samples thus taking a very long time. 
	  - Considering looking into exponential shadow mapping as a nicer looking and faster way to achieve soft shadowing.
	 */
	float shadow = 0.0f;
		
	for (float y = -1.5f; y <= 1.5f; y += 1.0f)
		for (float x = -1.5f; x <= 1.5f; x += 1.0f)
			shadow += texture(shadowTex, tsShadow.xyz + vec3(shadowSinglePixel.x * x, shadowSinglePixel.y * y, 0.0f));
		
	return shadow / 16.0f;
}

void main(void)	{
	vec3  normal		= normalize(texture(normalTex, IN.texCoord).xyz * 2.0f - 1.0f);
	vec3  colour 		= texture(colourTex, IN.texCoord).xyz;
	float depth 		= texture(depthTex, IN.texCoord).x;
	
	vec4 hwsPos = clipToWorldTransform * (vec4(IN.texCoord.x * 2.0f - 1.0f, IN.texCoord.y * 2.0f - 1.0f, depth * 2.0f - 1.0f, 1.0f));
	vec3 wsPos  = hwsPos.xyz / hwsPos.w;
	
	
//Shadow Calculations
	vec4 shadowWsPos = vec4(wsPos + normal * 0.2f, 1.0f);
	
	float shadow = 1.0f;
	vec3 shadowCol = vec3(0.0f);
	if (shadowNum > 0)
	{
		shadow = 0.0f;	

		int i = shadowNum - 1;
		for (; i >= 0; i--)
		{
			vec4 hcsShadow = shadowTransform[i] * shadowWsPos;
			vec3 tsShadow = (hcsShadow.xyz / hcsShadow.w) * 0.5f + 0.5f;
			
			if (tsShadow.x >= 0.0f && tsShadow.x <= 1.0f
				&& tsShadow.y >= 0.0f && tsShadow.y <= 1.0f
				&& tsShadow.z >= -1.0f && tsShadow.z <= 1.0f)
			{
				shadow += shadowTest(tsShadow, hcsShadow.w, shadowTex[i], shadowTransform[i]);
				break;
			}
		}

		if (i < 0) shadow = 1.0f; //Outside all shadowmaps - just pretend it's not shadowed
		shadow = max(shadow, 0.0f);
	}
	
//Lighting Calculations
	vec3 viewDir 		= normalize(cameraPos - wsPos );
	vec3 halfDir 		= normalize(invLightDir + viewDir);
	float rFactor       = max(0.0, dot(halfDir , normal ));
	
	float dFactor       = max(0.0, dot(invLightDir , normal )) * shadow;
    float sFactor       = pow(rFactor , specularIntensity ) * shadow;
	   
//Output Final L Colours
	OutFrag = vec4(dFactor, sFactor, 0.0, 1.0f);
}