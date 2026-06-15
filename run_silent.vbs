' LOCO Detector Silent Launcher
' Double-click this file to start LOCO Detector silently.
' No console windows appear (frontend runs minimized).
' Reads configuration from root .env file.

Dim shell, fso, currentDir, envPath
Set fso = CreateObject("Scripting.FileSystemObject")
currentDir = fso.GetParentFolderName(WScript.ScriptFullName)
Set shell = CreateObject("WScript.Shell")

' --- Read .env configuration ---
Dim backendPort, frontendPort, viteApiBase
backendPort = "8011"
frontendPort = "5173"
viteApiBase = "http://127.0.0.1:8011"

envPath = currentDir & "\.env"
If fso.FileExists(envPath) Then
    Dim envFile, line, eqPos, key, value
    Set envFile = fso.OpenTextFile(envPath, 1)
    Do Until envFile.AtEndOfStream
        line = Trim(envFile.ReadLine)
        If Len(line) > 0 And Left(line, 1) <> "#" Then
            eqPos = InStr(line, "=")
            If eqPos > 0 Then
                key = Trim(Left(line, eqPos - 1))
                value = Trim(Mid(line, eqPos + 1))
                If key = "BACKEND_PORT" Then backendPort = value
                If key = "FRONTEND_PORT" Then frontendPort = value
                If key = "VITE_API_BASE" Then viteApiBase = value
            End If
        End If
    Loop
    envFile.Close
End If

' --- Sync VITE_API_BASE into frontend/.env ---
Dim frontendEnvPath, frontendEnvContent
frontendEnvPath = currentDir & "\frontend\.env"
frontendEnvContent = "VITE_API_BASE=" & viteApiBase
Dim outFile
Set outFile = fso.CreateTextFile(frontendEnvPath, True)
outFile.WriteLine frontendEnvContent
outFile.Close

' --- Step 1: Stop previous LOCO processes via tools\stop_servers.ps1 ---
shell.Run "powershell -NoProfile -ExecutionPolicy Bypass -File """ & currentDir & "\tools\stop_servers.ps1""", 0, True

' --- Step 2: Wait for ports to release ---
WScript.Sleep 2000

' --- Step 3: Start backend (completely hidden) ---
shell.Run "powershell -WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -Command ""& { Set-Location '" & currentDir & "'; .\venv\Scripts\python.exe app.py }""", 0, False

' --- Step 4: Wait for backend using PowerShell (up to 30 seconds) ---
Dim healthCheckCmd
healthCheckCmd = "powershell -WindowStyle Hidden -NoProfile -Command ""$i=0; while($i -lt 30) { try { $r=curl.exe -s http://127.0.0.1:" & backendPort & "/api/health; if($r -match 'ok') { exit 0 } } catch {}; Start-Sleep 1; $i++ }; exit 1"""
shell.Run healthCheckCmd, 0, True

' --- Step 5: Start frontend (minimized - Vite needs a console context on Windows) ---
shell.Run "cmd /c start /min ""LOCO-Frontend"" /D """ & currentDir & "\frontend"" npm run dev -- --port " & frontendPort & " --host 127.0.0.1", 0, False

' --- Step 6: Open browser ---
WScript.Sleep 3000
shell.Run "http://127.0.0.1:" & frontendPort, 1, False

Set shell = Nothing
Set fso = Nothing
