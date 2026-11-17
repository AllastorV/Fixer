[Setup]
AppId={{841C8C4A-B69A-437D-AC61-BE2663D39E9F}
AppName=FIXER OpenCL
AppVersion={#Version}
AppPublisher=Alp Cavas
DefaultDirName={commoncf64}\OFX\Plugins\FixerOpenCL.ofx.bundle
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
OutputBaseFilename=FIXER-{#Version}-win64-opencl-setup
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "{#PackageDir}\FixerOpenCL.ofx.bundle\Contents\Win64\FixerOpenCL.ofx"; DestDir: "{app}\Contents\Win64"; Flags: ignoreversion
Source: "{#PackageDir}\FixerOpenCL.ofx.bundle\Contents\Resources\*"; DestDir: "{app}\Contents\Resources"; Flags: ignoreversion recursesubdirs
Source: "{#PackageDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PackageDir}\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PackageDir}\USER-GUIDE.md"; DestDir: "{app}"; Flags: ignoreversion
