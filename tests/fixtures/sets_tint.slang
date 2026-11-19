// One triangle over the whole target, placed by push constants, that
// samples an image and tints it.
struct Params {
    float4 scale_offset;
}

[[vk::push_constant]]
ConstantBuffer<Params> params;

struct Tint {
    float4 colour;
}

[[vk::binding(1, 0)]]
ConstantBuffer<Tint> tint : register(b1, space0);

[[vk::binding(0, 1)]]
Texture2D source : register(t0, space1);

[[vk::binding(1, 1)]]
SamplerState sampling : register(s1, space1);

struct VertexOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
}

[shader("vertex")]
VertexOutput vertex_main(float2 position : ATTRIB0) {
    VertexOutput output;
    float2 placed = position * params.scale_offset.xy + params.scale_offset.zw;
    output.position = float4(placed, 0.0, 1.0);
    output.uv = position * 0.5 + 0.5;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    return float4(source.SampleLevel(sampling, input.uv, 0.0).rgb * tint.colour.rgb, 1.0);
}
