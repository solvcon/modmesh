#version 440

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 texcoord;

layout(std140, binding = 0) uniform buf
{
    mat4 mvp;
    vec4 color;
} ubuf;

layout(location = 0) out vec2 v_texcoord;

void main()
{
    v_texcoord = texcoord;
    gl_Position = ubuf.mvp * vec4(position, 1.0);
}
