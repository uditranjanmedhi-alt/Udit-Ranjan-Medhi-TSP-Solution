Add-Type -AssemblyName System.IO.Compression.FileSystem
$zipPath = 'C:\Users\uditr\.gemini\antigravity\brain\99e6dbe4-2819-41be-a3a3-6a98ba473b69\.user_uploaded\media_1793788681997_c0b42c9a.zip'
$destDir = 'D:\IMI\TERM 5\DDDM VS\TSP_Data'
if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir }

$z = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
foreach ($entry in $z.Entries) {
    Write-Output "Entry: $($entry.FullName) ($($entry.Length) bytes)"
    $cleanName = [System.IO.Path]::GetFileName($entry.FullName)
    if ($cleanName) {
        $targetFile = [System.IO.Path]::Combine($destDir, $cleanName)
        Write-Output "Extracting to: $targetFile"
        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $targetFile, $true)
    }
}
$z.Dispose()
Write-Output "Extraction complete."

