// Raster: one full-screen triangle, positions scaled/offset by push
// constants, vertex color interpolated to the fragment stage.
struct Params {
    float4 scale_offset;
}

[[vk::push_constant]]
ConstantBuffer<Params> params;

struct VertexInput {
    float2 position : ATTRIB0;
    float4 colour : ATTRIB1;
}

struct VertexOutput {
    float4 position : SV_Position;
    float4 colour : COLOR0;
}

[shader("vertex")]
VertexOutput vertex_main(VertexInput input) {
    VertexOutput output;
    float2 position = input.position * params.scale_offset.xy + params.scale_offset.zw;
    output.position = float4(position, 0.0, 1.0);
    output.colour = input.colour;
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    return input.colour;
}

[shader("fragment")]
void depth_main() {
}
