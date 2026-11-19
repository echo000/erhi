# Compiles the shader sources of the tests and the examples to the SPIR-V
# and DXIL checked in beside them. The crate itself compiles nothing: this
# is only how its own shaders were built, and needs `slangc`, `dxc` and
# `glslc` (from the Vulkan SDK, or on the search path).
#
# Slang (`name.slang`): a file with a ray-tracing stage in it becomes one
# library, `name.spv` and `name.dxil`. Any other file becomes
# `name.<entry>.spv` and `name.<entry>.dxil` for each `[shader("...")]`
# entry point it has. A file with no entry point is only included by others.
#
# HLSL (`name.hlsl`): the same names, from `// entry: library` or a line
# `// entry: <vertex|pixel|compute> <name>` for each entry point. Its
# SPIR-V has samplers shifted by 16, constant buffers by 32 and what is
# written by 48, which a set says back as its register offsets.
#
# GLSL (`name.vert`, `.frag`, `.comp`, `.rgen`, `.rmiss`, `.rchit`, `.rahit`,
# `.rint`): `name.<extension>.spv`, for Vulkan alone.
#
#     pwsh compile_shaders.ps1            everything
#     pwsh compile_shaders.ps1 life       the files whose name has `life` in it

param([string]$Only = "")

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

function Find-Tool([string]$name) {
    if ($env:VULKAN_SDK) {
        foreach ($candidate in @("Bin/$name.exe", "bin/$name")) {
            $path = Join-Path $env:VULKAN_SDK $candidate
            if (Test-Path $path) {
                return $path
            }
        }
    }
    return $name
}

$slangc = Find-Tool "slangc"
$dxc = Find-Tool "dxc"
$glslc = Find-Tool "glslc"

$profiles = @{ compute = "cs_6_6"; vertex = "vs_6_6"; fragment = "ps_6_6" }
$entryPattern = '\[shader\("(\w+)"\)\]\s*(?:\[[^\]]*\]\s*)*[\w<>]+\s+(\w+)\s*\('
$failed = 0
$built = 0

function Invoke-Tool([string]$tool, [string]$output, [string[]]$arguments) {
    $log = & $tool @arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED $output"
        Write-Host ($log | Out-String)
        $script:failed += 1
    } else {
        $script:built += 1
    }
}

function Invoke-Slang([string]$source, [string]$output, [string[]]$arguments) {
    Invoke-Tool $slangc $output (@($source) + $arguments + @("-o", $output))
}

$hlslProfiles = @{ vertex = "vs_6_0"; pixel = "ps_6_0"; compute = "cs_6_0" }
$hlslShifts = @(
    "-fvk-s-shift", "16", "all", "-fvk-b-shift", "32", "all", "-fvk-u-shift", "48", "all"
)
$spirv = @("-spirv", "-fspv-target-env=vulkan1.2")
$glslStages = @(".vert", ".frag", ".comp", ".rgen", ".rmiss", ".rchit", ".rahit", ".rint")

foreach ($directory in @("tests/fixtures", "examples/shaders")) {
    $all = Get-ChildItem (Join-Path $root $directory) -File | Sort-Object Name
    $all = $all | Where-Object { -not $Only -or $_.BaseName.Contains($Only) }

    foreach ($file in $all | Where-Object { $_.Extension -eq ".hlsl" }) {
        $text = Get-Content $file.FullName -Raw
        $stem = Join-Path $file.DirectoryName $file.BaseName
        if ($text -match '(?m)^// entry: library\s*$') {
            Invoke-Tool $dxc "$stem.dxil" @("-T", "lib_6_3", $file.FullName, "-Fo", "$stem.dxil")
            Invoke-Tool $dxc "$stem.spv" (
                @("-T", "lib_6_3", $file.FullName, "-Fo", "$stem.spv") + $spirv
            )
            continue
        }
        foreach ($entry in [regex]::Matches($text, '(?m)^// entry: (\w+) (\w+)\s*$')) {
            $name = $entry.Groups[2].Value
            $common = @("-T", $hlslProfiles[$entry.Groups[1].Value], "-E", $name, $file.FullName)
            Invoke-Tool $dxc "$stem.$name.dxil" ($common + @("-Fo", "$stem.$name.dxil"))
            Invoke-Tool $dxc "$stem.$name.spv" (
                $common + @("-Fo", "$stem.$name.spv") + $spirv + $hlslShifts
            )
        }
    }

    foreach ($file in $all | Where-Object { $glslStages -contains $_.Extension }) {
        $output = "$($file.FullName).spv"
        Invoke-Tool $glslc $output @("--target-env=vulkan1.2", $file.FullName, "-o", $output)
    }

    foreach ($file in $all | Where-Object { $_.Extension -eq ".slang" }) {
        $text = Get-Content $file.FullName -Raw
        $entries = [regex]::Matches($text, $entryPattern)
        if ($entries.Count -eq 0) {
            continue
        }
        $stem = Join-Path $file.DirectoryName $file.BaseName
        $library = $entries | Where-Object { -not $profiles.ContainsKey($_.Groups[1].Value) }
        if ($library) {
            Invoke-Slang $file.FullName "$stem.spv" @("-target", "spirv", "-profile", "spirv_1_4")
            Invoke-Slang $file.FullName "$stem.dxil" @("-target", "dxil", "-profile", "lib_6_6")
            continue
        }
        foreach ($entry in $entries) {
            $stage = $entry.Groups[1].Value
            $name = $entry.Groups[2].Value
            Invoke-Slang $file.FullName "$stem.$name.spv" @(
                "-target", "spirv", "-profile", "spirv_1_4", "-entry", $name
            )
            Invoke-Slang $file.FullName "$stem.$name.dxil" @(
                "-target", "dxil", "-profile", $profiles[$stage], "-entry", $name
            )
        }
    }
}

Write-Host "$built compiled, $failed failed"
exit $failed
