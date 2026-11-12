@echo off
net session >nul 2>&1
if errorlevel 1 (
  powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
  exit /b
)
set "SRC=%~dp0build\release\Fixer.ofx.bundle"
if exist "%~dp0Fixer.ofx.bundle\Contents\Win64\Fixer.ofx" set "SRC=%~dp0Fixer.ofx.bundle"
set "DST=C:\Program Files\Common Files\OFX\Plugins\Fixer.ofx.bundle"
if not exist "%SRC%\Contents\Win64\Fixer.ofx" (
  echo Build bulunamadi: once build.bat calistir.
  pause
  exit /b 1
)
set "OLD=C:\Program Files\Common Files\OFX\Plugins\FilmEmulation.ofx.bundle"
if exist "%OLD%" if not exist "%OLD%.old" move "%OLD%" "%OLD%.old" >nul && echo Eski FilmEmulation.ofx.bundle devre disi: %OLD%.old
set "OLD2=C:\Program Files\Common Files\OFX\Plugins\LATERNA.ofx.bundle"
if exist "%OLD2%" if not exist "%OLD2%.old" move "%OLD2%" "%OLD2%.old" >nul && echo Eski LATERNA.ofx.bundle devre disi: %OLD2%.old
robocopy "%SRC%" "%DST%" /MIR /NFL /NDL /NJH /NJS
if errorlevel 8 (
  echo Kopyalama basarisiz.
  pause
  exit /b 1
)
echo Kuruldu: %DST%
echo Resolve'u yeniden baslat. Efekt: Color sayfasi ^> OpenFX ^> FIXER ^> FIXER
pause
