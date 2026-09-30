$d = (Resolve-Path "$PSScriptRoot\..\..").Path
$r = Join-Path (Split-Path $d -Parent) "netseedf-release"
New-Item -ItemType Directory -Path "$r\src" -Force | Out-Null
Copy-Item -Path "$d\src\main","$d\src\build","$d\src\freeze" -Destination "$r\src" -Recurse -Force
Copy-Item -Path "$d\uv.lock" -Destination "$r\uv.lock" -Force
Copy-Item -Path "$d\pyproject.toml" -Destination "$r\pyproject.toml" -Force