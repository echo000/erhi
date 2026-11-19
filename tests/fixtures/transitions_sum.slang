// Writes into `sum` the first number of `counter` and a hundred.
[[vk::binding(0, 0)]]
StructuredBuffer<uint> counter : register(t0, space0);

[[vk::binding(1, 0)]]
RWStructuredBuffer<uint> sum : register(u1, space0);

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    sum[0] = counter[0] + 100u;
}
