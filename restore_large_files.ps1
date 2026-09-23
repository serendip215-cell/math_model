param([string]$Repo = $PSScriptRoot)

$ErrorActionPreference = 'Stop'
$manifestPath = Join-Path $Repo 'large_files_manifest.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
$buffer = New-Object byte[] (4MB)

foreach ($entry in $manifest) {
  $destination = Join-Path $Repo ($entry.path -replace '/', [IO.Path]::DirectorySeparatorChar)
  if (Test-Path -LiteralPath $destination) {
    $existing = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($existing -ne $entry.sha256) { throw "Existing file has wrong SHA256: $($entry.path)" }
    Write-Output "Already valid: $($entry.path)"
    continue
  }
  $temporary = $destination + '.reassembling'
  if (Test-Path -LiteralPath $temporary) { throw "Temporary file exists: $temporary" }
  $outputStream = [IO.File]::Create($temporary)
  try {
    foreach ($part in $entry.parts) {
      $partPath = Join-Path $Repo ($part -replace '/', [IO.Path]::DirectorySeparatorChar)
      $inputStream = [IO.File]::OpenRead($partPath)
      try {
        while (($read = $inputStream.Read($buffer, 0, $buffer.Length)) -gt 0) {
          $outputStream.Write($buffer, 0, $read)
        }
      } finally { $inputStream.Dispose() }
    }
  } finally { $outputStream.Dispose() }
  $result = Get-Item -LiteralPath $temporary
  $hash = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($result.Length -ne [long]$entry.bytes -or $hash -ne $entry.sha256) {
    throw "Reassembled file failed verification: $($entry.path)"
  }
  Move-Item -LiteralPath $temporary -Destination $destination
  Write-Output "Restored: $($entry.path)"
}
