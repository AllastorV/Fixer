@echo off
net session >nul 2>&1
if errorlevel 1 (
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
set "SRC=%~dp0build\release\FixerOpenCL.ofx.bundle"
if exist "%~dp0FixerOpenCL.ofx.bundle\Contents\Win64\FixerOpenCL.ofx" set "SRC=%~dp0FixerOpenCL.ofx.bundle"
set "DST=C:\Program Files\Common Files\OFX\Plugins\FixerOpenCL.ofx.bundle"
if not exist "%SRC%\Contents\Win64\FixerOpenCL.ofx" (
  echo Build bulunamadi: once build.bat calistir.
  pause
  exit /b 1
)
robocopy "%SRC%" "%DST%" /MIR /NFL /NDL /NJH /NJS
if errorlevel 8 (
  echo Kopyalama basarisiz.
  pause
  exit /b 1
)
echo Kuruldu: %DST%
echo Resolve'u yeniden baslat. GPU processing mode: OpenCL. Efekt: Color sayfasi ^> OpenFX ^> FIXER ^> FIXER OpenCL
pause
