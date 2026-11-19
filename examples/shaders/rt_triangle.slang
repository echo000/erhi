// The ray-traced triangle example: one ray a pixel, the triangle coloured by
// where on it the ray lands, a dusk sky where it lands on nothing.

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure scene : register(t0, space0);

[format("rgba8")] [[vk::binding(1, 0)]]
RWTexture2D<float4> target : register(u1, space0);

struct Payload {
    float4 colour;
}

[shader("raygeneration")]
void ray_gen() {
    uint2 pixel = DispatchRaysIndex().xy;
    float2 size = float2(DispatchRaysDimensions().xy);
    // From -1 to 1 up the window, and as far across as the window is wide.
    float2 at = (float2(pixel) + 0.5) / size * 2.0 - 1.0;
    at.x *= size.x / size.y;

    RayDesc ray;
    ray.Origin = float3(0.0, 0.0, -2.5);
    ray.Direction = normalize(float3(at.x, -at.y, 1.5));
    ray.TMin = 0.001;
    ray.TMax = 100.0;

    Payload payload;
    payload.colour = float4(0.0, 0.0, 0.0, 1.0);
    TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);
    target[pixel] = payload.colour;
}

[shader("miss")]
void miss(inout Payload payload) {
    float up = WorldRayDirection().y * 0.5 + 0.5;
    payload.colour = float4(lerp(float3(0.03, 0.04, 0.07), float3(0.10, 0.13, 0.22), up), 1.0);
}

[shader("closesthit")]
void closest_hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    float2 b = attributes.barycentrics;
    payload.colour = float4(1.0 - b.x - b.y, b.x, b.y, 1.0);
}
