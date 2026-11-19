struct VertexOutput {
    float4 position : SV_Position;
}

[shader("vertex")]
VertexOutput vertex_main(float2 position : ATTRIB0) {
    VertexOutput output;
    output.position = float4(position, 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragment_main(VertexOutput input) : SV_Target {
    return float4(0.0, 1.0, 0.0, 1.0);
}
