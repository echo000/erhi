// The triangle example's two entry points.

struct Params {
    // The turn as (cosine, sine), then the window's height over its width.
    float4 turn_aspect;
}

[[vk::push_constant]]
ConstantBuffer<Params> params;

struct VertexInput {
    float2 position : ATTRIB0;
    float3 colour : ATTRIB1;
}

struct VertexOutput {
    float4 position : SV_Position;
    float3 colour : COLOR0;
}

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    float2 turn = params.turn_aspect.xy;
    float2 turned = float2(input.position.x * turn.x - input.position.y * turn.y,
                           input.position.x * turn.y + input.position.y * turn.x);
    VertexOutput output;
    output.position = float4(turned.x * params.turn_aspect.z, turned.y, 0.0, 1.0);
    output.colour = input.colour;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    return float4(input.colour, 1.0);
}
