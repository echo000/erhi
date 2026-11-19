// Compute: sample the source through the layout's immutable linear-clamp
// sampler at texel centers (an identity read), add a push-constant color,
// and write the result to the storage image.
struct Params {
    float4 add;
}

[[vk::binding(0, 0)]]
Sampler2D source_image : register(t0, space0);

[format("rgba8")] [[vk::binding(1, 0)]]
RWTexture2D<float4> destination_image : register(u1, space0);

[[vk::push_constant]]
ConstantBuffer<Params> params;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 thread_id : SV_DispatchThreadID) {
    uint2 id = thread_id.xy;
    float2 uv = (float2(id) + float2(0.5, 0.5)) / 8.0;
    float4 colour = source_image.SampleLevel(uv, 0.0);
    destination_image[int2(id)] = clamp(colour + params.add, 0.0, 1.0);
}
