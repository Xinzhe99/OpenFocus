; OpenFocus Windows installer (Inno Setup 6)
; Built in CI with: ISCC.exe /DAPP_VERSION=<tag> OpenFocus.iss

#define MyAppName "OpenFocus"
#ifndef APP_VERSION
#define APP_VERSION "dev"
#endif
#ifndef APP_VERSION_NUM
#define APP_VERSION_NUM "0.0.0"
#endif
#define MyAppVersion APP_VERSION
#define MyAppExeName "OpenFocus.exe"

[Setup]
AppId={{8C1F4E52-9B7A-4A2E-9C1D-3F5A7B9E1D42}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
VersionInfoCompany=OpenFocus Contributors
VersionInfoDescription={#MyAppName} installer
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#APP_VERSION_NUM}
VersionInfoVersion={#APP_VERSION_NUM}
VersionInfoTextVersion={#APP_VERSION_NUM}
VersionInfoProductTextVersion={#APP_VERSION_NUM}
VersionInfoCopyright=MIT License
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
OutputDir=installer
OutputBaseFilename=OpenFocus-{#MyAppVersion}-setup
SetupIconFile=assets\OpenFocus.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "explorercontext"; Description: "Add OpenFocus to the Windows Explorer context menu"

[InstallDelete]
; These directories belong to the installed application; keep user data intact.
Type: filesandordirs; Name: "{app}\_internal"; Check: FileExists(ExpandConstant('{app}\OpenFocus.exe'))
Type: filesandordirs; Name: "{app}\_openfocus_runtime_*"; Check: FileExists(ExpandConstant('{app}\OpenFocus.exe'))

[Files]
Source: "dist\OpenFocus\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; The unique verb and CLSID are the only keys owned and removed by this
; installer. Existing file associations and Explorer settings are untouched.
Root: HKA; Subkey: "Software\Classes\CLSID\{{08A1D5C2-DB59-4A62-AC2B-ED9E960A986E}"; Flags: deletekey; Check: not WizardIsTaskSelected('explorercontext')
Root: HKA; Subkey: "Software\Classes\Directory\shell\OpenFocus.Import"; Flags: deletekey; Check: not WizardIsTaskSelected('explorercontext')
Root: HKA; Subkey: "Software\Classes\*\shell\OpenFocus.Import"; Flags: deletekey; Check: not WizardIsTaskSelected('explorercontext')
Root: HKA; Subkey: "Software\Classes\CLSID\{{08A1D5C2-DB59-4A62-AC2B-ED9E960A986E}"; ValueType: string; ValueData: "OpenFocus Explorer import"; Flags: uninsdeletekey; Tasks: explorercontext
Root: HKA; Subkey: "Software\Classes\CLSID\{{08A1D5C2-DB59-4A62-AC2B-ED9E960A986E}\LocalServer32"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" --shell-server"; Tasks: explorercontext
Root: HKA; Subkey: "Software\Classes\Directory\shell\OpenFocus.Import"; ValueType: string; ValueData: "Import into OpenFocus"; Flags: uninsdeletekey; Tasks: explorercontext
Root: HKA; Subkey: "Software\Classes\Directory\shell\OpenFocus.Import"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#MyAppExeName},0"; Tasks: explorercontext
Root: HKA; Subkey: "Software\Classes\Directory\shell\OpenFocus.Import"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Player"; Tasks: explorercontext
Root: HKA; Subkey: "Software\Classes\Directory\shell\OpenFocus.Import\DropTarget"; ValueType: string; ValueName: "Clsid"; ValueData: "{{08A1D5C2-DB59-4A62-AC2B-ED9E960A986E}"; Tasks: explorercontext
Root: HKA; Subkey: "Software\Classes\*\shell\OpenFocus.Import"; ValueType: string; ValueData: "Import into OpenFocus"; Flags: uninsdeletekey; Tasks: explorercontext
Root: HKA; Subkey: "Software\Classes\*\shell\OpenFocus.Import"; ValueType: string; ValueName: "Icon"; ValueData: "{app}\{#MyAppExeName},0"; Tasks: explorercontext
Root: HKA; Subkey: "Software\Classes\*\shell\OpenFocus.Import"; ValueType: string; ValueName: "MultiSelectModel"; ValueData: "Player"; Tasks: explorercontext
Root: HKA; Subkey: "Software\Classes\*\shell\OpenFocus.Import"; ValueType: string; ValueName: "AppliesTo"; ValueData: "System.FileName:""*.jpg"" OR System.FileName:""*.jpeg"" OR System.FileName:""*.jpe"" OR System.FileName:""*.jfif"" OR System.FileName:""*.png"" OR System.FileName:""*.bmp"" OR System.FileName:""*.tiff"" OR System.FileName:""*.tif"" OR System.FileName:""*.webp"" OR System.FileName:""*.heic"" OR System.FileName:""*.heif"" OR System.FileName:""*.3fr"" OR System.FileName:""*.arw"" OR System.FileName:""*.bay"" OR System.FileName:""*.cap"" OR System.FileName:""*.cr2"" OR System.FileName:""*.cr3"" OR System.FileName:""*.crw"" OR System.FileName:""*.dc2"" OR System.FileName:""*.dcr"" OR System.FileName:""*.dng"" OR System.FileName:""*.erf"" OR System.FileName:""*.fff"" OR System.FileName:""*.iiq"" OR System.FileName:""*.k25"" OR System.FileName:""*.kc2"" OR System.FileName:""*.kdc"" OR System.FileName:""*.mdc"" OR System.FileName:""*.mef"" OR System.FileName:""*.mos"" OR System.FileName:""*.mrw"" OR System.FileName:""*.nef"" OR System.FileName:""*.nrw"" OR System.FileName:""*.orf"" OR System.FileName:""*.pef"" OR System.FileName:""*.raf"" OR System.FileName:""*.raw"" OR System.FileName:""*.rdc"" OR System.FileName:""*.rw2"" OR System.FileName:""*.rwl"" OR System.FileName:""*.srf"" OR System.FileName:""*.sr2"" OR System.FileName:""*.srw"" OR System.FileName:""*.sti"" OR System.FileName:""*.x3f"""; Tasks: explorercontext
Root: HKA; Subkey: "Software\Classes\*\shell\OpenFocus.Import\DropTarget"; ValueType: string; ValueName: "Clsid"; ValueData: "{{08A1D5C2-DB59-4A62-AC2B-ED9E960A986E}"; Tasks: explorercontext

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent
