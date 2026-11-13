[Setup]
AppId={{9D729058-FD18-44DE-80DA-264D81219190}
AppName=FIXER
AppVersion={#Version}
AppPublisher=Alp Cavas
DefaultDirName={commoncf64}\OFX\Plugins\Fixer.ofx.bundle
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=admin
ArchitecturesAllowed=x64os
ArchitecturesInstallIn64BitMode=x64os
OutputBaseFilename=FIXER-{#Version}-win64-cuda-setup
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern

[Files]
Source: "{#PackageDir}\Fixer.ofx.bundle\Contents\Win64\Fixer.ofx"; DestDir: "{app}\Contents\Win64"; Flags: ignoreversion
Source: "{#PackageDir}\Fixer.ofx.bundle\Contents\Resources\*"; DestDir: "{app}\Contents\Resources"; Flags: ignoreversion recursesubdirs
Source: "{#PackageDir}\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PackageDir}\NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#PackageDir}\USER-GUIDE.md"; DestDir: "{app}"; Flags: ignoreversion
