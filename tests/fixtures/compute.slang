// Minimal real compute entry for module creation and validation tests.
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
}
