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

struct ShiftedInput {
    float2 position : ATTRIB0;
    float4 colour : ATTRIB1;
    float2 shift : ATTRIB2;
}

[shader("vertex")]
VertexOutput vertex_shifted(ShiftedInput input) {
    VertexOutput output;
    output.position = float4(input.position + input.shift, 0.0, 1.0);
    output.colour = input.colour;
    return output;
}

// One triangle over the whole viewport, from the index of each vertex alone.
[shader("vertex")]
VertexOutput vertex_bare(uint index : SV_VertexID) {
    VertexOutput output;
    float2 corner = float2(float((index << 1) & 2u), float(index & 2u));
    output.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);
    output.colour = float4(0.0, 1.0, 0.0, 1.0);
    return output;
}

struct Two {
    float4 first : SV_Target0;
    float4 second : SV_Target1;
}

[shader("fragment")]
Two fragment_two(VertexOutput input) {
    Two output;
    output.first = input.colour;
    output.second = float4(0.25, 0.5, 0.75, 1.0);
    return output;
}
