// Texels 0 to 5: the six faces of the cube, each read along its own axis.
// Texels 6 and 7: the mipped image at its largest level and far past its
// smallest, through the sampler's level-of-detail range.
[[vk::binding(0, 0)]]
Texture2D<float4> mips : register(t0, space0);

[[vk::binding(1, 0)]]
SamplerState point_sampler : register(s1, space0);

[[vk::binding(2, 0)]]
TextureCube<float4> cube : register(t2, space0);

[format("rgba8")] [[vk::binding(3, 0)]]
RWTexture2D<float4> destination : register(u3, space0);

[shader("compute")]
[numthreads(8, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float3 directions[6] = {float3(1, 0, 0),  float3(-1, 0, 0), float3(0, 1, 0),
                            float3(0, -1, 0), float3(0, 0, 1),  float3(0, 0, -1)};
    float4 colour;
    if (id.x < 6u) {
        colour = cube.SampleLevel(point_sampler, directions[id.x], 0.0);
    } else {
        colour = mips.SampleLevel(point_sampler, float2(0.5, 0.5), id.x == 6u ? 0.0 : 5.0);
    }
    destination[int2(id.xy)] = colour;
}
