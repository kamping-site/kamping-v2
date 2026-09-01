#!/bin/bash
# Standalone SLURM job for the isolated grid_alltoallv size-transition data-loss
# reproducer. See KaCCv2's notes/grid-alltoallv-supermuc-data-loss.md for the full
# investigation writeup this belongs to.
#
# Runs TEST(GridAlltoallvTest, SizeTransitionPoisonRepro)
# (dstl/tests/grid_alltoallv_test.cpp, this branch, commit cca2920) -- pure MPI +
# dstl::grid_alltoallv, no KaCCv2/graph/BFS involved. Replays the real repro's exact
# per-round size sequence, poison-fills each receive buffer with 0xAA before the real
# MPI_Alltoallv call, and fails (with stderr detail) if any element still shows the
# poison pattern afterward.
#
# Assumes test_dstl_grid_alltoallv is already built -- this script only runs it. Point
# BIN at your build tree if it isn't at the default path below.
#
# SBATCH header copied from a known-working KaCCv2 jobfile
# (frozen-multistep-kout-supermuc-grid-fanout-diagnostic_impi-2021-17, 2026-09-01) --
# same partition/account/switches/node count as every other job in this investigation.
#
#SBATCH --nodes=2
#SBATCH -o %x-%j.log.txt
#SBATCH -e %x-%j.err.txt
#SBATCH -J grid-alltoallv-size-transition-repro
#SBATCH --partition=test
#SBATCH --time=0-00:12:00
#SBATCH --get-user-env
#SBATCH --export=NONE
#SBATCH --account=pn72pu
#SBATCH --switches=1
#SBATCH --ear=off

set -uo pipefail
# Deliberately no -e: a test FAILURE (the bug reproducing) makes gtest exit non-zero,
# and we want the "Finished ... in N seconds" line below to still print in that case.

# Path to the already-built binary. Default assumes a repo-root-relative build/release
# tree (this repo's own CMakePresets.json "release" preset); override with
# BIN=/path/to/test_dstl_grid_alltoallv sbatch ... if yours lives elsewhere (e.g. under
# KaCCv2's build-dev-supermuc/_deps/kamping-v2-src/<your-build-dir>).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$REPO_ROOT/build/release/dstl/tests/test_dstl_grid_alltoallv}"

if [ ! -x "$BIN" ]; then
  echo "[LOG] ERROR: no executable at $BIN -- build test_dstl_grid_alltoallv first, or set BIN=..."
  exit 1
fi

# Same module set as the other jobs in this investigation -- change this if you want to
# test a different Intel MPI version (matches the impi-2021-17 / impi-2021-15 runs
# already in KaCCv2's eval/data/supermuc/).
module restore kaccv2-impi-2021.17
module load slurm_setup

islands_used=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n 's/^\(i[0-9][0-9]*\).*/\1/p' | sort -u | paste -sd, -)
echo "[LOG] JobID=${SLURM_JOB_ID}"
echo "[LOG] Nodes=${SLURM_JOB_NODELIST}"
echo "[LOG] Islands=${islands_used:-unknown}"
echo "[LOG] NumNodes=${SLURM_JOB_NUM_NODES}"
echo "[LOG] Modules:"
module list 2>&1
echo "[LOG] PATH=$PATH"
echo "[LOG] BIN=$BIN"

echo "[LOG] Starting grid-alltoallv-size-transition-repro"
start=$(date +%s.%N)
I_MPI_PIN_CELL=core I_MPI_PIN_DOMAIN=1:compact I_MPI_JOB_TIMEOUT=120 KACC_REPRO_ITERATIONS=5 \
  mpiexec -n 2 --perhost 1 "$BIN" --gtest_filter='*SizeTransitionPoisonRepro*'
end=$(date +%s.%N)
diff=$(echo "$end - $start" | bc)
echo "[LOG] Finished grid-alltoallv-size-transition-repro in ${diff} seconds"
