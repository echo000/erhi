// Conway's Game of Life on the compute pipeline. A cell is one texel: red
// says whether it lives, green how lately it did.

struct Params {
    uint seed;
    uint3 unused;
}

[[vk::push_constant]]
ConstantBuffer<Params> params;

[[vk::binding(0, 0)]]
Texture2D<float4> current : register(t0, space0);

[format("rgba8")] [[vk::binding(1, 0)]]
RWTexture2D<float4> next : register(u1, space0);

uint hash(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

// A third of the cells alive, chosen by `params.seed`.
[shader("compute")]
[numthreads(8, 8, 1)]
void seed(uint3 id : SV_DispatchThreadID) {
    uint2 size;
    next.GetDimensions(size.x, size.y);
    if (any(id.xy >= size)) {
        return;
    }
    uint drawn = hash(id.x + hash(id.y + hash(params.seed)));
    float alive = drawn % 3 == 0 ? 1.0 : 0.0;
    next[id.xy] = float4(alive, alive, 0.0, 1.0);
}

// One generation: `next` from `current`, the board joined edge to edge.
[shader("compute")]
[numthreads(8, 8, 1)]
void advance(uint3 id : SV_DispatchThreadID) {
    uint2 size;
    next.GetDimensions(size.x, size.y);
    if (any(id.xy >= size)) {
        return;
    }
    uint neighbours = 0;
    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) {
                continue;
            }
            uint2 at = uint2((int2(id.xy) + int2(x, y) + int2(size)) % int2(size));
            neighbours += current.Load(int3(at, 0)).r > 0.5 ? 1 : 0;
        }
    }
    float4 cell = current.Load(int3(id.xy, 0));
    bool lives = cell.r > 0.5 ? (neighbours == 2 || neighbours == 3) : neighbours == 3;
    next[id.xy] = float4(lives ? 1.0 : 0.0, lives ? 1.0 : cell.g * 0.93, 0.0, 1.0);
}

// The board drawn over the whole of `next`, whatever size that is.
[shader("compute")]
[numthreads(8, 8, 1)]
void show(uint3 id : SV_DispatchThreadID) {
    uint2 size;
    next.GetDimensions(size.x, size.y);
    if (any(id.xy >= size)) {
        return;
    }
    uint2 board;
    current.GetDimensions(board.x, board.y);
    float4 cell = current.Load(int3(id.xy * board / size, 0));
    float3 colour = lerp(float3(0.03, 0.04, 0.07), float3(0.10, 0.35, 0.55), cell.g);
    colour = lerp(colour, float3(0.95, 1.00, 0.85), cell.r);
    next[id.xy] = float4(colour, 1.0);
}
