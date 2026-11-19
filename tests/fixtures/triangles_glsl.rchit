#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT vec4 colour;

void main() {
    colour = vec4(gl_PrimitiveID == 0 ? 1.0 : 0.0, 1.0, 0.0, 1.0);
}
