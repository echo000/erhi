// The ray-traced scene example: a knot over a chequered floor, lit by two
// lights, with shadows and reflections that are rays of their own.

static const uint LIGHTS = 2;

struct View {
    float4 eye;
    float4 forward;
    float4 right;
    float4 up;
    // Where each light is, and in `w` how bright.
    float4 lights[LIGHTS];
    // What colour each light is.
    float4 colours[LIGHTS];
}

struct Vertex {
    float4 position;
    float4 normal;
}

[[vk::binding(0, 0)]]
RaytracingAccelerationStructure scene : register(t0, space0);

[format("rgba8")] [[vk::binding(1, 0)]]
RWTexture2D<float4> target : register(u1, space0);

[[vk::binding(2, 0)]]
ConstantBuffer<View> view : register(b2, space0);

[[vk::binding(3, 0)]]
StructuredBuffer<Vertex> vertices : register(t3, space0);

[[vk::binding(4, 0)]]
StructuredBuffer<uint> indices : register(t4, space0);

// The floor is the first two triangles of the mesh.
static const uint FLOOR_TRIANGLES = 2;

static const uint BOUNCES = 3;

// What a ray into the scene comes back with: where it landed and on what.
struct Payload {
    float3 position;
    // Negative where it landed on nothing.
    float distance;
    float3 normal;
    // 0 for the floor, 1 for the knot.
    float surface;
}

// What a ray towards the light comes back with.
struct Shadow {
    float lit;
}

float3 sky(float3 direction) {
    float up = saturate(direction.y * 0.5 + 0.5);
    return lerp(float3(0.20, 0.22, 0.28), float3(0.35, 0.50, 0.85), up);
}

[shader("raygeneration")]
void ray_gen() {
    uint2 pixel = DispatchRaysIndex().xy;
    float2 size = float2(DispatchRaysDimensions().xy);
    float2 at = (float2(pixel) + 0.5) / size * 2.0 - 1.0;
    at.x *= size.x / size.y;

    RayDesc ray;
    ray.Origin = view.eye.xyz;
    ray.Direction =
        normalize(view.forward.xyz + at.x * 0.6 * view.right.xyz - at.y * 0.6 * view.up.xyz);
    ray.TMin = 0.001;
    ray.TMax = 1000.0;

    float3 colour = float3(0.0, 0.0, 0.0);
    float weight = 1.0;
    for (uint bounce = 0; bounce < BOUNCES; bounce++) {
        Payload payload;
        payload.distance = -1.0;
        TraceRay(scene, RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, payload);
        if (payload.distance < 0.0) {
            colour += weight * sky(ray.Direction);
            break;
        }

        float3 albedo = float3(0.90, 0.55, 0.20);
        float mirror = 0.12;
        if (payload.surface < 0.5) {
            float2 cell = floor(payload.position.xz);
            float dark = abs(fmod(cell.x + cell.y, 2.0));
            albedo = lerp(float3(0.75, 0.75, 0.72), float3(0.18, 0.19, 0.22), dark);
            mirror = 0.30;
        }

        // Is each light seen from here? A ray towards it only has to land
        // on anything, so the first hit ends it and no hit shader runs.
        float3 received = float3(0.06, 0.06, 0.06);
        for (uint light = 0; light < LIGHTS; light++) {
            float3 to_light = view.lights[light].xyz - payload.position;
            float reach = length(to_light);
            RayDesc towards;
            towards.Origin = payload.position + payload.normal * 0.002;
            towards.Direction = to_light / reach;
            towards.TMin = 0.0;
            towards.TMax = reach;
            Shadow shadow;
            shadow.lit = 0.0;
            TraceRay(scene,
                     RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
                     0xff, 0, 0, 1, towards, shadow);

            float facing = saturate(dot(payload.normal, towards.Direction));
            received += view.colours[light].rgb *
                        (shadow.lit * facing * view.lights[light].w / (reach * reach));
        }
        colour += weight * (1.0 - mirror) * albedo * received;

        weight *= mirror;
        ray.Origin = payload.position + payload.normal * 0.002;
        ray.Direction = reflect(ray.Direction, payload.normal);
    }

    // The image holds what a screen shows, not light: bring the range in
    // and encode it.
    colour = colour / (1.0 + colour);
    target[pixel] = float4(pow(colour, 1.0 / 2.2), 1.0);
}

[shader("miss")]
void miss(inout Payload payload) {
    payload.distance = -1.0;
}

[shader("miss")]
void shadow_miss(inout Shadow shadow) {
    shadow.lit = 1.0;
}

[shader("closesthit")]
void closest_hit(inout Payload payload, BuiltInTriangleIntersectionAttributes attributes) {
    uint first = PrimitiveIndex() * 3;
    float3 weights = float3(1.0 - attributes.barycentrics.x - attributes.barycentrics.y,
                            attributes.barycentrics.x, attributes.barycentrics.y);
    float3 normal = normalize(vertices[indices[first]].normal.xyz * weights.x +
                              vertices[indices[first + 1]].normal.xyz * weights.y +
                              vertices[indices[first + 2]].normal.xyz * weights.z);
    // Towards whoever is looking.
    if (dot(normal, WorldRayDirection()) > 0.0) {
        normal = -normal;
    }
    payload.position = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    payload.distance = RayTCurrent();
    payload.normal = normal;
    payload.surface = PrimitiveIndex() < FLOOR_TRIANGLES ? 0.0 : 1.0;
}
