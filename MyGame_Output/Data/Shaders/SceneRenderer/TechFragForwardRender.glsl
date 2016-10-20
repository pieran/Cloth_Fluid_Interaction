#version 150 core

uniform sampler2D diffuseTex;

uniform vec3  	invLightDir;			//Directional Light
uniform vec3  	cameraPos;
uniform float 	specularIntensity;
uniform vec3  	ambientColour;

uniform int 			shadowNum;
uniform sampler2DShadow shadowTex[16];
uniform mat4 			shadowTransform[16];
uniform vec2 			shadowSinglePixel = vec2(1.0f / 2048.0f);

in Vertex	{
	vec4 worldPos;
	vec2 texCoord;
	vec4 colour;
	vec3 normal;
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
	vec3 normal 		= normalize(IN.normal);
	vec4 texColour  	= texture(diffuseTex, IN.texCoord);
	vec4 colour 		= IN.colour * texColour;
	vec3 wsPos 			= IN.worldPos.xyz / IN.worldPos.w;
	
	//Shadow Calculations
	vec4 shadowWsPos = vec4(wsPos + normal * 0.2f, 1.0f);
	
	
	float shadow = 1.0f;
	if (shadowNum > 0)
	{
		shadow = 0.0f;
		for (int i = shadowNum - 1; i >= 0; i--)
		{
			vec4 hcsShadow = shadowTransform[i] * shadowWsPos;
			vec3 tsShadow = (hcsShadow.xyz / hcsShadow.w) * 0.5f + 0.5f;
			
			if (tsShadow.x >= 0.0f && tsShadow.x <= 1.0f
				&& tsShadow.y >= 0.0f && tsShadow.y <= 1.0f)
			{
				shadow += shadowTest(tsShadow, hcsShadow.w, shadowTex[i], shadowTransform[i]);
				break;
			}
		}

		shadow = max(shadow, 0.0f);
	}
	
//Lighting Calculations
	vec3 viewDir 		= normalize(cameraPos - wsPos );
	vec3 halfDir 		= normalize(invLightDir + viewDir);
	float rFactor       = max(0.0, dot(halfDir , normal ));
	
	float dFactor       = max(0.0, dot(invLightDir , normal )) ;
    float sFactor       = pow(rFactor , specularIntensity );
	   
//Colour Computations
	vec3 specColour = min(colour.rgb + vec3(0.5f), vec3(1)); //Quick hack to approximate specular colour of an object, assuming the light colour is white
	
//Output Final Colour
	vec3 diffuse = colour.rgb * dFactor * shadow;
	vec3 specular = specColour * sFactor * shadow;
	
	OutFrag.xyz 	= colour.rgb * ambientColour + (diffuse + specular * 0.5f) * (vec3(1) - ambientColour);
	OutFrag.a 		= colour.a;
}