// Reads three buffers of one binding of set 0 and one of set 1, and writes
// what they add up to through set 2.
[[vk::binding(0, 0)]]
StructuredBuffer<uint> digits[3] : register(t0, space0);

[[vk::binding(0, 1)]]
StructuredBuffer<uint> scale : register(t0, space1);

[[vk::binding(0, 2)]]
RWStructuredBuffer<uint> sum : register(u0, space2);

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    sum[0] = (digits[0][0] + digits[1][0] * 10u + digits[2][0] * 100u) * scale[0];
}
