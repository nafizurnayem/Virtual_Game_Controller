; Inno Setup installer for Virtual Steering Wheel
; Requires Inno Setup 6 on the Windows build machine.
#define AppName "Virtual Steering Wheel"
#define AppVersion "1.0.0"
#define AppPublisher "Virtual Steering Wheel"
#define AppExeName "VirtualSteeringWheel.exe"

[Setup]
AppId={{D5A0E761-2A11-4D4B-A2B3-3F5E7E5F38D8}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\Virtual Steering Wheel
DefaultGroupName={#AppName}
OutputDir=installer
OutputBaseFilename=VirtualSteeringWheel-Setup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64
PrivilegesRequired=admin
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExeName}

[Files]
Source: "..\dist\{#AppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\README.md"; DestDir: "{app}"; Flags: isreadme

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Run]
Filename: "{app}\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent runasoriginaluser

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
