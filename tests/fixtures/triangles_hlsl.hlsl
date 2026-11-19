// entry: library
//
// What `triangles_pipeline.slang` is, in HLSL: one library for `dxc`.
[[vk::binding(0, 0)]]
RaytracingAccelerationStructure scene : register(t0, space0);

[[vk::image_format("rgba8")]]
[[vk::binding(1, 0)]]
RWTexture2D<float4> target : register(u1, space0);

struct Payload {
    float4 colour;
};

[shader("raygeneration")]
void ray_gen() {
    uint2 pixel = DispatchRaysIndex().xy;
    float2 at = (float2(pixel) + 0.5) / 16.0 * 2.0 - 1.0;
    RayDesc ray;
    ray.Origin = float3(at, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 10.0;
    Payload payload;
    payload.colour = float4(0.0, 0.0, 0.0, 0.0);
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);
    target[pixel] = payload.colour;
}

[shader("miss")]
void miss(inout Payload payload) {
    payload.colour = float4(0.0, 0.0, 1.0, 1.0);
}

[shader("closesthit")]
void closest_hit(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes) {
    payload.colour = float4(PrimitiveIndex() == 0 ? 1.0 : 0.0, 1.0, 0.0, 1.0);
}

[shader("anyhit")]
void let_through(inout Payload payload, in BuiltInTriangleIntersectionAttributes attributes) {
    IgnoreHit();
}
