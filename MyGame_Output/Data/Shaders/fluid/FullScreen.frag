#version 150 core

uniform sampler2D texNormals;
uniform sampler2D texAbsorbtion;
uniform sampler2D texDepth;
uniform samplerCube texReflect;
uniform sampler2D texRefract;

uniform vec3  ambientColour;
uniform vec3  invLightDir;			//Directional Light
uniform vec3  cameraPos;
uniform float specularIntensity = 128.0f;
uniform float baseReflect = 0.1f;
//uniform float baseRefract = 0.8f;

uniform vec3 absorbtionExp = vec3(45.5f, 25.2f, 8.1f);
uniform mat4 invProjViewMatrix;
uniform mat3 viewMatrix;

in Vertex	{
	vec2 texCoords;
} IN;

out vec4 gl_FragColor;


void main(void)	{
	float ndc_depth = texture(texDepth, IN.texCoords).x;
	if (ndc_depth > 0.9999f)
	{
		discard;
		return;
	}
	
	//Set Fragment Depth
	gl_FragDepth = ndc_depth;
	
	
	//Read Texture Specific Data
	vec3 normal = normalize(texture(texNormals, IN.texCoords).rgb * 2.0f - 1.0f);
	vec3 normalEs = normalize(viewMatrix * normal);
	float absorbtion = texture(texAbsorbtion, IN.texCoords).x;
	absorbtion = min(absorbtion * 0.5f, 0.1f);
	
	//Compute World-Space Position
	vec4 clip_pos = vec4(IN.texCoords.x, IN.texCoords.y, ndc_depth, 1.0f) * 2.0f - 1.0f;
	vec4 ws_pos = invProjViewMatrix * clip_pos;
	ws_pos.xyz /= ws_pos.w;
	
	
	
	//Lighting Calculations (Diffuse + Specular Factors)
	vec3 viewDir 		= normalize(cameraPos - ws_pos.xyz );
	vec3 halfDir 		= normalize(invLightDir + viewDir );
	float rFactor       = max(0.0, dot(halfDir , normal ));
	
	float dFactor       = max(0.0, dot(invLightDir , normal )) ;
    float sFactor       = pow(rFactor , specularIntensity );
	   
	//Lighting Calculations (Fresnel) 
	float theta = clamp(dot(viewDir, normal), 0.0f, 1.0f);
	float fFactor = 0.0f;//baseReflect + (1.0f - baseReflect) * pow(1.0f - theta, 2.0f);  
	//fFactor = clamp(fFactor, baseReflect, 0.1f);

	//Calculate Base Color (from absorbtion)
	vec3 diffColour = vec3(1.0f);
	diffColour.r = clamp(exp(-absorbtionExp.r * absorbtion), 0, 1);
	diffColour.g = clamp(exp(-absorbtionExp.g * absorbtion), 0, 1);
	diffColour.b = clamp(exp(-absorbtionExp.b * absorbtion), 0, 1);
	
	
	float ratio = 1.0 /1.3333; //refractive index
	vec3 refractCoords = refract(-viewDir, normal, ratio);
	vec3 viewDir2 		= -normalize(ws_pos.xyz - cameraPos);
	vec3 reflColour = vec3(0.8f);//texture(texReflect, normalize(reflect(-viewDir, normal))).rgb;
	//vec3 refrColour = texture(texReflect, refractCoords).rgb;// - absorbtion * normal * 0.1f).rgb;
	
	vec3 refrColour = texture(texRefract, IN.texCoords + absorbtion * normalEs.xy).rgb;
	
	
	float refractExp = mix(absorbtionExp.g, absorbtionExp.b, 0.9f);
	float baseRefract = clamp(exp(-refractExp * absorbtion), 0.0f, 1.0f);	
	diffColour *= (1.0f - baseRefract);
	diffColour += refrColour * baseRefract;
	diffColour *= (1.0f - fFactor);
	diffColour += reflColour * fFactor;
//diffColour = refrColour;
	
	//Colour Computations
	vec3 specColour = min(diffColour + vec3(0.5f), vec3(1)); //Quick hack to approximate specular colour of an object, assuming the light colour is white
    vec3 col = ambientColour * diffColour 
			 + diffColour * dFactor
			 + specColour * sFactor * 0.6;
		
		
	//Output Final Fragment Colour
	gl_FragColor.rgb 	= col; //normal * 0.5f + 0.5f;//
	float factor = clamp(absorbtion * 100.0f,  0.0f, 1.0f);
	factor = pow(factor, 15.0f);
	gl_FragColor.a 		= clamp(factor, 0.05f, 1.0f);
	//gl_FragColor.rgb = normalEs.xyz * 0.5f + 0.5f;
	//gl_FragColor.r = absorbtion;
}