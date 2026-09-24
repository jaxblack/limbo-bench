"""Build the arXiv source package from the TMLR manuscript.

    ..\\.venv\\Scripts\\python make_arxiv.py

Produces arxiv_build/ (a clean copy with the preprint option and author block,
compiled once to obtain main.bbl, which arXiv needs because it does not run
BibTeX) and arxiv_build/limbo_arxiv_source.tar.gz containing only the files
arXiv needs to rebuild the PDF.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "arxiv_build"
TECTONIC = Path(os.environ.get("LOCALAPPDATA", "")) / "limbo-tools" / "tectonic.exe"

AUTHOR_BLOCK = r"""\author{\name Jiapeng Li \email jiapengli@microsoft.com \\
      \addr Microsoft}"""
CODE_REPO = "https://github.com/jaxblack/limbo-bench"
CODE_MACROS = (rf"\newcommand{{\CodeURL}}{{\url{{{CODE_REPO}}}}}" "\n"
               r"\newcommand{\CodeCite}{~\citep{li2026limbo}}" "\n")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "sections").mkdir(parents=True)
    (OUT / "generated").mkdir()
    main_tex = (HERE / "main.tex").read_text(encoding="utf-8")
    main_tex = main_tex.replace(r"\usepackage{tmlr}", r"\usepackage[preprint]{tmlr}", 1)
    main_tex, n = re.subn(r"\\author\{.*?\}\s*\n", lambda m: AUTHOR_BLOCK + "\n", main_tex, count=1, flags=re.S)
    assert n == 1, "author block not found"
    main_tex, n = re.subn(r"\\newcommand\{\\CodeURL\}\{.*?\}\n\\newcommand\{\\CodeCite\}\{\}\n",
                          lambda m: CODE_MACROS, main_tex, count=1)
    assert n == 1, "code release macros not found"
    (OUT / "main.tex").write_text(main_tex, encoding="utf-8")
    for name in ("tmlr.sty", "fancyhdr.sty", "tmlr.bst", "math_commands.tex", "references.bib"):
        shutil.copy2(HERE / name, OUT / name)
    for f in (HERE / "sections").glob("*.tex"):
        shutil.copy2(f, OUT / "sections" / f.name)
    used = set(re.findall(r"generated/([\w.\-]+)", " ".join(
        p.read_text(encoding="utf-8") for p in [OUT / "main.tex", *(OUT / "sections").glob("*.tex")])))
    for name in sorted(used):
        src = HERE / "generated" / name
        if not src.exists() and (HERE / "generated" / f"{name}.tex").exists():
            src = HERE / "generated" / f"{name}.tex"
        shutil.copy2(src, OUT / "generated" / src.name)
    subprocess.run([str(TECTONIC), "--keep-intermediates", "--keep-logs", "main.tex"], cwd=OUT, check=True,
                   capture_output=True)
    assert (OUT / "main.bbl").exists(), "main.bbl was not produced"
    # arXiv flags text files without a final newline as possibly truncated.
    for p in OUT.rglob("*"):
        if p.suffix in (".tex", ".bbl", ".sty") and p.is_file():
            data = p.read_bytes()
            if data and not data.endswith(b"\n"):
                p.write_bytes(data + b"\n")
    members = ["main.tex", "main.bbl", "tmlr.sty", "tmlr.bst", "fancyhdr.sty", "math_commands.tex"]
    members += [f"sections/{p.name}" for p in sorted((OUT / "sections").glob("*.tex"))]
    members += [f"generated/{p.name}" for p in sorted((OUT / "generated").iterdir())]
    tar_path = OUT / "limbo_arxiv_source.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for m in members:
            tar.add(OUT / m, arcname=m)
    print(f"package: {tar_path} ({tar_path.stat().st_size / 1024:.0f} KiB, {len(members)} files)")
    print(f"preview PDF: {OUT / 'main.pdf'}")


if __name__ == "__main__":
    main()
