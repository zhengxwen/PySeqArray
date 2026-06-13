#!/usr/bin/env bash
#
# sync_seqarray.sh — pull a new SeqArray upstream into PySeqArray's vendored tree
# and show exactly what changed, so the option-B native rewrite only has to be
# re-applied to the hunks that actually moved.
#
# Layout:
#   seqarray/upstream/   pristine baseline (NEVER hand-edited) — for diffing
#   seqarray/src/        PySeqArray working copy (the native-rewritten engine)
#
# Usage:
#   tools/sync_seqarray.sh /path/to/SeqArray/src
#
# Workflow on an upstream update:
#   1. Run this with the new SeqArray src path.
#   2. It prints the upstream diff (old baseline -> new) per file.
#   3. Re-apply the same SEXP->PyObject transform to just those hunks in
#      seqarray/src/, then rebuild + run tests.
#   4. The baseline is refreshed to the new upstream automatically.

set -euo pipefail

HERE="$(cd "$(dirname "$0")/.." && pwd)"
UP="$HERE/seqarray/upstream"
SRC="$HERE/seqarray/src"

NEW="${1:-}"
if [[ -z "$NEW" || ! -d "$NEW" ]]; then
    echo "usage: $0 /path/to/SeqArray/src" >&2
    exit 1
fi

echo "== Upstream diff (vendored baseline -> $NEW) =="
changed=0
for f in "$NEW"/*.cpp "$NEW"/*.h "$NEW"/*.c; do
    [[ -e "$f" ]] || continue
    b="$UP/$(basename "$f")"
    if [[ ! -e "$b" ]]; then
        echo "NEW FILE: $(basename "$f")"
        changed=1
    elif ! diff -q "$b" "$f" >/dev/null 2>&1; then
        echo "--- CHANGED: $(basename "$f") ---"
        diff -u "$b" "$f" || true
        changed=1
    fi
done
[[ "$changed" -eq 0 ]] && echo "(no upstream changes)"

echo
echo "== Refreshing pristine baseline -> seqarray/upstream/ =="
cp "$NEW"/*.cpp "$NEW"/*.h "$NEW"/*.c "$UP"/ 2>/dev/null || true
if command -v git >/dev/null && [[ -d "$NEW/../.git" ]]; then
    ( cd "$NEW/.." && git rev-parse HEAD ) > "$HERE/seqarray/UPSTREAM_VERSION.txt" 2>/dev/null || true
fi

echo
echo "Next: apply the printed changes to seqarray/src/ (the native-rewritten copy),"
echo "then:  cd $HERE && python3 setup.py build_ext --inplace && python3 tests/test_engine.py"
