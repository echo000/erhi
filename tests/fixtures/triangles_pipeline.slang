// Red where the first triangle of a structure is hit, green wherever any
// is; blue where nothing is. The second hit group takes no hit at all.

#include "triangles_common.slang"

struct Payload {
    float4 colour;
}

[shader("raygeneration")]
void ray_gen() {
    uint2 pixel = DispatchRaysIndex().xy;
    Payload payload;
    payload.colour = float4(0.0, 0.0, 0.0, 0.0);
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray_through(pixel), payload);
    target[pixel] = payload.colour;
}

[shader("miss")]
void miss(inout Payload payload) {
    payload.colour = float4(0.0, 0.0, 1.0, 1.0);
}

[shader("closesthit")]
void closest_hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    payload.colour = float4(PrimitiveIndex() == 0 ? 1.0 : 0.0, 1.0, 0.0, 1.0);
}

[shader("anyhit")]
void let_through(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    IgnoreHit();
}
