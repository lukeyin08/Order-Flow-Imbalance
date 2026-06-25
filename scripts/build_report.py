#!/usr/bin/env python3
"""Compile the report from its LaTeX source to report/OFI_Project_Report.pdf.

Requires a LaTeX install (pdflatex, e.g. MacTeX or TeX Live). The compiled PDF is
committed to the repo, so LaTeX is only needed if you want to rebuild it.

    python scripts/build_report.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(ROOT, "report")
TEX = "OFI_Project_Report.tex"


def main():
    if shutil.which("pdflatex") is None:
        sys.exit("pdflatex not found. Install a LaTeX distribution (MacTeX or TeX "
                 "Live) to rebuild the report. The compiled PDF is already in report/.")
    for _ in range(2):  # twice so references/labels resolve
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", TEX],
            cwd=REPORT_DIR)
        if result.returncode != 0:
            sys.exit("pdflatex failed; see report/OFI_Project_Report.log")
    for ext in (".aux", ".log", ".out", ".fls", ".fdb_latexmk"):
        aux = os.path.join(REPORT_DIR, "OFI_Project_Report" + ext)
        try:
            if os.path.exists(aux):
                os.remove(aux)
        except OSError:
            pass
    print("report ->", os.path.join(REPORT_DIR, "OFI_Project_Report.pdf"))


if __name__ == "__main__":
    main()
