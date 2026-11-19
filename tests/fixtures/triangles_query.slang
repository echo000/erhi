// The same colours from a query, with blue added where the hit was of
// geometry that is not opaque (which a query has to take by hand).

#include "triangles_common.slang"

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    RayQuery<RAY_FLAG_NONE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray_through(id.xy));
    float asked = 0.0;
    while (query.Proceed()) {
        if (query.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE) {
            asked = 1.0;
            query.CommitNonOpaqueTriangleHit();
        }
    }
    float4 colour = float4(0.0, 0.0, 1.0, 1.0);
    if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT) {
        colour = float4(query.CommittedPrimitiveIndex() == 0 ? 1.0 : 0.0, 1.0, asked, 1.0);
    }
    target[id.xy] = colour;
}
