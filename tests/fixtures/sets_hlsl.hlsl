// entry: vertex vertex_main
// entry: pixel pixel_main
//
// HLSL written the Direct3D way: each kind of register counted from zero,
// inputs by semantics of their own, the push constants at a register that
// is said. `dxc` compiles it to DXIL as it stands and to SPIR-V with each
// kind of register shifted, which the set's register offsets undo.
struct Params {
    float4 scale_offset;
};

[[vk::push_constant]]
ConstantBuffer<Params> params : register(b1, space0);

cbuffer Tint : register(b0, space0) {
    float4 tint_colour;
};

Texture2D source : register(t0, space0);

SamplerState sampling : register(s0, space0);

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

VertexOutput vertex_main(float2 position : POSITION) {
    VertexOutput output;
    float2 placed = position * params.scale_offset.xy + params.scale_offset.zw;
    output.position = float4(placed, 0.0, 1.0);
    output.uv = position * 0.5 + 0.5;
    return output;
}

float4 pixel_main(VertexOutput input) : SV_Target {
    return float4(source.SampleLevel(sampling, input.uv, 0.0).rgb * tint_colour.rgb, 1.0);
}
