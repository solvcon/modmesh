#version 440

layout(location = 0) in vec2 v_texcoord;

layout(binding = 1) uniform sampler2D tex;

layout(std140, binding = 0) uniform buf
{
    mat4 mvp;
    vec4 color;
} ubuf;

layout(location = 0) out vec4 fragColor;

void main()
{
    // The label texture already carries the axis color and the glyph coverage
    // in its alpha; the uniform color is a global multiplier (white).
    fragColor = texture(tex, v_texcoord) * ubuf.color;
}
