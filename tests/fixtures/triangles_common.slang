// What both tracers share: the scene, the target, and one ray a pixel,
// straight along z through the square from -1 to 1.
[[vk::binding(0, 0)]]
RaytracingAccelerationStructure scene : register(t0, space0);

[format("rgba8")] [[vk::binding(1, 0)]]
RWTexture2D<float4> target : register(u1, space0);

RayDesc ray_through(uint2 pixel) {
    float2 at = (float2(pixel) + 0.5) / 16.0 * 2.0 - 1.0;
    RayDesc ray;
    ray.Origin = float3(at, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 10.0;
    return ray;
}
