#version 330

uniform uint objID;

layout(location = 0, index = 0) out uint OutFrag;

void main(void)	{
	OutFrag   = objID;
}