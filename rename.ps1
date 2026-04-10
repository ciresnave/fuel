$exts = @('.rs', '.md', '.toml', '.txt', '.yml', '.yaml', '.ps1', '.sh', '.html', '.c', '.cpp', '.h', '.cu')
$files = git ls-files
foreach ($f in $files) {
    $ext = [System.IO.Path]::GetExtension($f).ToLower()
    if ($exts -contains $ext) {
        $path = "$pwd\$f"
        if (Test-Path $path) {
            try {
                $content = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
                if ($content -match '(?i)fuel') {
                    $content = $content -creplace 'fuel', 'fuel' -creplace 'Fuel', 'Fuel' -creplace 'FUEL', 'FUEL'
                    [System.IO.File]::WriteAllText($path, $content, [System.Text.Encoding]::UTF8)
                }
            } catch {
                Write-Host "Skipped text replace on $path"
            }
        }
    }
}

Get-ChildItem -Recurse | Where-Object { $_.FullName -notmatch '\\\.git\\' -and $_.Name -match '(?i)fuel' } | Sort-Object -Property @{Expression={$_.FullName.Length}; Descending=$true} | ForEach-Object {
    $newName = $_.Name -creplace 'fuel', 'fuel' -creplace 'Fuel', 'Fuel' -creplace 'FUEL', 'FUEL'
    Rename-Item -Path $_.FullName -NewName $newName
}

git add .
git status
