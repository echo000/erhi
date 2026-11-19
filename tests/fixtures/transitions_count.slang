// Adds one to the first number of `counter`.
[[vk::binding(0, 0)]]
RWStructuredBuffer<uint> counter : register(u0, space0);

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    counter[0] += 1u;
}
