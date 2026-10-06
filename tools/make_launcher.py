"""Create Myna.exe: a real interpreter named after the app, with its icon.

Two problems this solves:

* A venv's ``pythonw.exe`` on a conda base is a redirector, so the process that
  actually runs is the base interpreter and Task Manager shows "pythonw.exe"
  with the base install's path. Copying a real interpreter into the venv under
  our own name makes Task Manager show "Myna.exe".
* That copy still carries Python's icon, which Explorer and Task Manager show.
  The icon is replaced in the copy's resource table.

Run: python tools/make_launcher.py <venv_scripts_dir> <icon.ico>
"""

from __future__ import annotations

import ctypes
import shutil
import struct
import sys
import sysconfig
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.branding import LAUNCHER_EXE  # noqa: E402


def _find_real_interpreter(scripts_dir: Path) -> Path:
    """The genuine pythonw.exe from the base installation.

    The copy in a venv's Scripts cannot be trusted: on a conda base it is a
    small redirector that re-execs the base interpreter, so a copy of it still
    shows up as the base interpreter in Task Manager. Size is not a reliable
    tell - that stub is 244 KB, larger than the real 88 KB interpreter - so
    take the base installation's copy unconditionally.

    A real interpreter placed in Scripts still finds ../pyvenv.cfg and uses the
    venv's site-packages, which is exactly how an ordinary venv works.
    """
    base = Path(sys.base_prefix)
    for name in ("pythonw.exe", "python.exe"):
        found = base / name
        if found.is_file():
            return found
    raise FileNotFoundError(
        f"Could not locate a real pythonw.exe under {base}."
    )


def _copy_python_dlls(base: Path, scripts_dir: Path) -> None:
    """Put python3X.dll beside the copied interpreter.

    Windows resolves an exe's DLL imports before Python runs, from the exe's
    own folder and PATH. Without this the copy only starts when the base
    installation happens to be on PATH, and fails with "python3X.dll was not
    found" as soon as it is not.
    """
    names = [f"python{sys.version_info[0]}{sys.version_info[1]}.dll", "python3.dll"]
    for name in names:
        found = base / name
        if found.is_file():
            shutil.copy2(found, scripts_dir / name)
            print(f"Copied {name} beside {LAUNCHER_EXE}")


def _write_conda_dll_path(base: Path) -> None:
    """Let extension modules find a conda base's DLLs (OpenSSL, libffi, sqlite).

    A conda interpreter only adds <prefix>\\Library\\bin for itself, and in a venv
    the prefix is the venv. A .pth file runs at every start-up and adds the base
    installation's folder instead. Not needed (and skipped) on a python.org base.
    """
    library_bin = base / "Library" / "bin"
    if not library_bin.is_dir():
        return
    site_packages = Path(sysconfig.get_paths()["purelib"])
    pth = site_packages / "myna_conda_dlls.pth"
    pth.write_text(
        "import os, sys; p = os.path.join(sys.base_prefix, 'Library', 'bin'); "
        "os.path.isdir(p) and os.add_dll_directory(p)\n",
        encoding="ascii",
    )
    print(f"Wrote {pth.name} so extension modules find {library_bin}")


# ---------------------------------------------------------------- icon swap
RT_ICON = 3
RT_GROUP_ICON = 14


def _icon_resources(ico_path: Path) -> tuple[bytes, list[bytes]]:
    """Split an .ico into a GROUP_ICON directory and its image entries."""
    data = ico_path.read_bytes()
    reserved, ico_type, count = struct.unpack("<HHH", data[:6])
    if reserved != 0 or ico_type != 1:
        raise ValueError("Not a valid .ico file")

    entries, images = [], []
    for index in range(count):
        offset = 6 + index * 16
        width, height, colours, res, planes, bits, size, image_offset = struct.unpack(
            "<BBBBHHII", data[offset : offset + 16]
        )
        images.append(data[image_offset : image_offset + size])
        # GROUP_ICON entries replace the file offset with a resource id.
        entries.append(
            struct.pack(
                "<BBBBHHIH", width, height, colours, res, planes, bits, size, index + 1
            )
        )

    group = struct.pack("<HHH", 0, 1, count) + b"".join(entries)
    return group, images


def set_exe_icon(exe_path: Path, ico_path: Path) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.BeginUpdateResourceW.restype = wintypes.HANDLE
    kernel32.BeginUpdateResourceW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL]
    kernel32.UpdateResourceW.argtypes = [
        wintypes.HANDLE, wintypes.LPCWSTR, wintypes.LPCWSTR,
        wintypes.WORD, wintypes.LPVOID, wintypes.DWORD,
    ]
    kernel32.EndUpdateResourceW.argtypes = [wintypes.HANDLE, wintypes.BOOL]

    group, images = _icon_resources(ico_path)

    handle = kernel32.BeginUpdateResourceW(str(exe_path), False)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        for index, image in enumerate(images, start=1):
            if not kernel32.UpdateResourceW(
                handle, wintypes.LPCWSTR(RT_ICON), wintypes.LPCWSTR(index),
                0, image, len(image),
            ):
                raise ctypes.WinError(ctypes.get_last_error())

        # Resource id 1 is the group Explorer shows for the file.
        if not kernel32.UpdateResourceW(
            handle, wintypes.LPCWSTR(RT_GROUP_ICON), wintypes.LPCWSTR(1),
            0, group, len(group),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
    except Exception:
        kernel32.EndUpdateResourceW(handle, True)  # discard
        raise

    if not kernel32.EndUpdateResourceW(handle, False):
        raise ctypes.WinError(ctypes.get_last_error())


def main() -> int:
    scripts_dir = Path(sys.argv[1])
    icon = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    target = scripts_dir / LAUNCHER_EXE

    source = _find_real_interpreter(scripts_dir)
    shutil.copy2(source, target)
    print(f"Created {target.name} from {source}")
    _copy_python_dlls(source.parent, scripts_dir)
    _write_conda_dll_path(source.parent)

    if icon and icon.is_file():
        try:
            set_exe_icon(target, icon)
            print(f"Embedded {icon.name} into {target.name}")
        except Exception as exc:  # noqa: BLE001 - cosmetic, never fatal
            print(f"Could not embed the icon ({exc}); shortcuts still carry it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
