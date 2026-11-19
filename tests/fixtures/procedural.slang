// Test fixture: a ray-tracing pipeline over procedural (box) geometry.
struct Payload {
    float4 color;
};

struct Attributes {
    float2 uv;
};

[shader("raygeneration")]
void ray_gen() {
}

[shader("miss")]
void primary_miss(inout Payload p) {
    p.color = float4(0, 0, 0, 1);
}

[shader("miss")]
void shadow_miss(inout Payload p) {
    p.color = float4(1, 1, 1, 1);
}

[shader("intersection")]
void intersection() {
    Attributes a;
    a.uv = 0;
    ReportHit(1.0, 0, a);
}

[shader("closesthit")]
void closest_hit(inout Payload p, Attributes a) {
    p.color = float4(a.uv, 0, 1);
}
