#!/usr/bin/env python3
"""Build a CPU quantization bridge from a pinned llama.cpp revision.

Downloads only GGML base sources/headers (no model weights or GPU backend).
Runtime never downloads or compiles implicitly. Re-run this script on each host.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import urllib.request

REVISION = "ceefd16daa2093730bf3340315fa2b84b286ddbc"
BASE_SOURCES = ["ggml.c", "ggml.cpp", "ggml-alloc.c", "ggml-backend.cpp",
                "ggml-backend-meta.cpp", "ggml-opt.cpp", "ggml-threading.cpp",
                "ggml-quants.c", "gguf.cpp"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/brmoe-ggml" / REVISION)
    parser.add_argument("--no-proxy", action="store_true", help="Bypass environment proxy for source download")
    args = parser.parse_args()
    root = args.cache_dir.resolve()
    src = root / "source"
    src.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({})) if args.no_proxy else urllib.request.build_opener()

    def fetch(url):
        with opener.open(url, timeout=45) as response:
            return response.read()

    tree = json.loads(fetch(f"https://api.github.com/repos/ggml-org/llama.cpp/git/trees/{REVISION}?recursive=1"))
    if tree.get("truncated"):
        raise RuntimeError("GitHub source tree is truncated")
    paths = [t["path"] for t in tree["tree"] if t["type"] == "blob" and (
        (t["path"].startswith("ggml/include/") and t["path"].endswith(".h")) or
        (t["path"].startswith("ggml/src/") and t["path"].count("/") == 2 and t["path"].endswith(".h")) or
        (t["path"].startswith("ggml/src/ggml-cpu/") and t["path"].count("/") == 3 and t["path"].endswith(".h")) or
        t["path"] in [f"ggml/src/{name}" for name in BASE_SOURCES] or t["path"] == "LICENSE")]

    def download(path):
        out = src / path
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            data = fetch(f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{REVISION}/{path}")
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(out)

    print(f"Fetching {len(paths)} source files at {REVISION}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(download, paths))
    (src / "ggml/src/ggml-version.h").write_text(
        f'#define GGML_VERSION "brmoe-{REVISION[:12]}"\n#define GGML_COMMIT "{REVISION}"\n')
    bridge = Path(__file__).resolve().parents[1] / "brmoe_quant/bridge.c"
    source_list = "\n".join(f'"{src / "ggml/src" / name}"' for name in BASE_SOURCES)
    (root / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.16)
project(brmoe_ggml_quant LANGUAGES C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
find_package(Threads REQUIRED)
add_library(brmoe_ggml_quant SHARED {source_list} "{bridge}")
target_include_directories(brmoe_ggml_quant PRIVATE "{src / 'ggml/include'}" "{src / 'ggml/src'}")
target_compile_definitions(brmoe_ggml_quant PRIVATE GGML_BUILD GGML_SHARED)
target_link_libraries(brmoe_ggml_quant PRIVATE Threads::Threads ${{CMAKE_DL_LIBS}})
''')
    subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build"), "-DCMAKE_BUILD_TYPE=Release"], check=True)
    subprocess.run(["cmake", "--build", str(root / "build"), "--parallel", "4"], check=True)
    print(f"Built native GGML bridge in {root / 'build'}", flush=True)


if __name__ == "__main__":
    main()
