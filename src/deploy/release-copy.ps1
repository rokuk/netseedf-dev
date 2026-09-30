$d="C:\Users\rokku\Documents\Code\netseedf-dev";
$r="C:\Users\rokku\Documents\Code\netseedf-release";
New-Item -ItemType Directory -Path $r -Force | Out-Null;
Copy-Item -Path "$d\src\main","$d\src\build" -Destination "$r\src" -Recurse -Force
Copy-Item -Path "$d\uv.lock" -Destination "$r\uv.lock" -Force
Copy-Item -Path "$d\pyproject.toml" -Destination "$r\pyproject.toml" -Force