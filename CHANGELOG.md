# Changelog

What changes between versions, newest first. Until 1.0 any release may
break the API; what breaks is said here.

## 0.1.0

First public version.

- Vulkan and Direct3D 12 behind one API, chosen when a device is opened.
- Render, compute and ray-tracing pipelines; acceleration structures of
  triangles and of boxes; ray queries from compute.
- Shaders as compiled SPIR-V or DXIL, whatever they were compiled from:
  Slang, HLSL and GLSL all run in the tests.
- Resource states tracked for the caller, a mip and a layer at a time.
- Every refusal an error; nothing printed; debug-layer messages handed to
  the caller.
- Examples: a window, a triangle, a swap of backends while running,
  Conway's Game of Life on compute, a ray-traced triangle and a ray-traced
  scene.
