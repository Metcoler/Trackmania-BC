from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent
LOCAL_OUTPUT = PLUGIN_DIR / "get_data_driver.op"
OPENPLANET_OUTPUT = Path(r"C:\Users\sampu\OpenplanetNext\Plugins\get_data_driver.op")


def build_plugin_archive(output_path: Path) -> None:
    with zipfile.ZipFile(output_path, "w") as archive:
        archive.write(PLUGIN_DIR / "info.toml", "info.toml")
        archive.write(PLUGIN_DIR / "main.as", "main.as")


def main() -> None:
    build_plugin_archive(LOCAL_OUTPUT)
    print(f"Built local plugin archive: {LOCAL_OUTPUT}")

    try:
        OPENPLANET_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LOCAL_OUTPUT, OPENPLANET_OUTPUT)
        print(f"Installed plugin archive: {OPENPLANET_OUTPUT}")
    except PermissionError as exc:
        print(
            "Could not overwrite the OpenPlanet plugin archive. "
            "Trackmania/OpenPlanet is probably using it right now."
        )
        print(f"Local archive is ready to copy manually after closing Trackmania: {LOCAL_OUTPUT}")
        print(f"Details: {exc}")


if __name__ == "__main__":
    main()
