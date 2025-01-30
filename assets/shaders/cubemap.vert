#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    float texture_weight;
} ubo;

layout(location = 0) in vec3 vPosition;

layout(location = 0) out vec3 fragDir;

void main() {
    fragDir = vPosition;
    gl_Position = ubo.proj * ubo.view * vec4(vPosition, 1.0);
}
