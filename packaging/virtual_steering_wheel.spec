# PyInstaller build specification for Windows 10/11 x64.
# Run with: pyinstaller --clean --noconfirm virtual_steering_wheel.spec
from PyInstaller.utils.hooks import collect_all


datas = []
binaries = []
hiddenimports = []

for package in ("mediapipe", "cv2", "numpy", "pynput"):
    # A missing package must fail the build instead of producing an EXE that
    # installs successfully but crashes when the camera starts.
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# OpenCV and MediaPipe can contain duplicate entries when collected together.
def unique(items):
    seen = set()
    result = []
    for item in items:
        key = tuple(item) if isinstance(item, (list, tuple)) else item
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


a = Analysis(
    ["../src/steering_wheel.py"],
    pathex=["."],
    binaries=unique(binaries),
    datas=unique(datas),
    hiddenimports=unique(hiddenimports),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="VirtualSteeringWheel",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
)
