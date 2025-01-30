#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float texture_weight;
} ubo;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragCoords;

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) out vec4 outColor;

// from <https://stackoverflow.com/a/10625698>
float random(vec2 p) {
    vec2 K1 = vec2(
        23.14069263277926, // e^pi
        2.665144142690225  // 2^sqrt(2)
    );
    return fract(cos(dot(p, K1)) * 12345.6789);
}

void main() {
    vec4 color = vec4(
        random(vec2(gl_PrimitiveID, 1.1)),
        random(vec2(gl_PrimitiveID, 2.2)),
        random(vec2(gl_PrimitiveID, 3.3)),
        1.0
    );
    vec4 tex = texture(texSampler, fragCoords);
    outColor = mix(color, tex, ubo.texture_weight);
}
