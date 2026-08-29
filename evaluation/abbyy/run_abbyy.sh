#!/usr/bin/env bash
# Compile and run the ABBYY FineReader Engine 12 (Linux, Java) layout runner
# over our 860-page test set and write YOLO dumps for run_literature_eval.py.
#
# Fill in the four FRE_* variables after you install the trial SDK (or export
# them in your shell). Everything ABBYY-specific is provided by ABBYY together
# with the trial licence.
#
#   FRE_HOME            root of the FineReader Engine install (has Bin/, Inc/, Samples/)
#   FRE_CUSTOMER_ID     "Customer project ID" string from ABBYY (online licence)
#   FRE_LICENSE_PATH    path to the licence file ABBYY sent (often FRE_HOME/Bin.<arch>/license)
#                       -- leave "" for a pure online licence
#   FRE_LICENSE_PW      licence password from ABBYY
#
# Usage:
#   ./run_abbyy.sh                       # full 860-page run
#   ./run_abbyy.sh 5                     # smoke test on 5 pages
set -euo pipefail

FRE_HOME="${FRE_HOME:-/opt/ABBYY/FREngine12}"
FRE_CUSTOMER_ID="${FRE_CUSTOMER_ID:-}"
FRE_LICENSE_PATH="${FRE_LICENSE_PATH:-}"
FRE_LICENSE_PW="${FRE_LICENSE_PW:-}"

# On Linux the native binaries live in a Bin.<arch> folder; the Java wrapper jar
# (FREngine.jar) usually sits under Inc/Java. Adjust if your layout differs.
FRE_BIN="${FRE_BIN:-$FRE_HOME/Bin.x64}"
FRE_JAR="${FRE_JAR:-$FRE_HOME/Inc/Java/FREngine.jar}"

IMG_DIR="${IMG_DIR:-/home/eroux/azure_di_eval/testset/images/test}"
OUT_DIR="${OUT_DIR:-/home/eroux/abbyy_eval}"
MAX_PAGES="${1:-0}"

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "FRE_HOME   = $FRE_HOME"
echo "FRE_BIN    = $FRE_BIN"
echo "FRE_JAR    = $FRE_JAR"
[ -f "$FRE_JAR" ] || { echo "ERROR: FREngine.jar not found at $FRE_JAR (find it under $FRE_HOME and set FRE_JAR)"; exit 1; }
[ -d "$FRE_BIN" ] || { echo "ERROR: FRE bin dir not found at $FRE_BIN (set FRE_BIN)"; exit 1; }
[ -n "$FRE_CUSTOMER_ID" ] || { echo "ERROR: set FRE_CUSTOMER_ID (from ABBYY)"; exit 1; }

echo "== compiling =="
javac -cp "$FRE_JAR" -d "$HERE" "$HERE/AbbyyLayoutToYolo.java"

echo "== running (maxPages=$MAX_PAGES) =="
# FRE native libs must be on the loader path.
export LD_LIBRARY_PATH="$FRE_BIN:${LD_LIBRARY_PATH:-}"
java -cp "$FRE_JAR:$HERE" AbbyyLayoutToYolo \
    "$FRE_BIN" \
    "$FRE_CUSTOMER_ID" \
    "$FRE_LICENSE_PATH" \
    "$FRE_LICENSE_PW" \
    "$IMG_DIR" \
    "$OUT_DIR" \
    "$MAX_PAGES"

echo "== dumps in $OUT_DIR/labels =="
