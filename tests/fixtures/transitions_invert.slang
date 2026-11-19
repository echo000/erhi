// Writes into `destination` what `source` is not.
[[vk::binding(0, 0)]]
Sampler2D source : register(t0, space0);

[format("rgba8")] [[vk::binding(1, 0)]]
RWTexture2D<float4> destination : register(u1, space0);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float4 texel = source.SampleLevel((float2(id.xy) + 0.5) / 16.0, 0.0);
    destination[id.xy] = float4(1.0 - texel.rgb, 1.0);
}
