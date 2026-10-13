#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT

# Run the pic-agentic simulation-side client on a cluster login node.
#
# It clones the repo, installs it into a local venv, performs the
# once-interactive MAS device login (scripts/mas_login.py) and then runs the
# simclient. The simclient holds the Matrix control channel and, on an
# `rcp.hello` command from the MCP server, submits a trivial job with the real
# `sbatch`/`scontrol` from PATH. Runs entirely in user space; needs no sudo.
#
# Usage (on the login node):
#
#   export PIC_AGENTIC_ROOM_ID='!....:academiccloud.de'
#   export PIC_AGENTIC_RCP_SECRET='<64 hex chars>'
#   bash cluster_simclient.sh
#
# Environment (ROOM_ID and RCP_SECRET are required):
#   PIC_AGENTIC_ROOM_ID        Matrix room id (required)
#   PIC_AGENTIC_RCP_SECRET     shared per-simulation HMAC secret (required)
#   PIC_AGENTIC_HOMESERVER     default https://chat.academiccloud.de
#   PIC_AGENTIC_SIM            simulation id, default "cluster"
#   PIC_AGENTIC_BRANCH         git branch/tag to install, default "main"
#   PIC_AGENTIC_WORKDIR        default "$HOME/pic-agentic"
#   PIC_AGENTIC_MESSAGE_DIR    shared dir for message files, default "$WORKDIR/shared"
#   PIC_AGENTIC_SIM_SETUP_ROOT shared dir for generated setups, default
#                              "$WORKDIR/sims" (enables the M2 submit handler)
#   PIC_AGENTIC_CLUSTER_TEMPLATE_DIR  cluster-local picongpu template dir
#   PIC_AGENTIC_CLUSTER_PRESET cluster-local CMake configure preset number
#   PIC_AGENTIC_PICONGPU_REVISION     pinned picongpu revision (drift check)
#   PIC_AGENTIC_JOB_WAIT_TIMEOUT_S  default 600 (queue waits)
#   PIC_AGENTIC_ACK_TIMEOUT_S  default 900
#   PIC_AGENTIC_POLL_INTERVAL_S     initial watcher poll interval, default 30
#   PIC_AGENTIC_POLL_MAX_INTERVAL_S watcher backoff cap, default 300
#   PIC_AGENTIC_SKIP_LOGIN=1   reuse an existing token config
#   PIC_AGENTIC_NO_UPDATE=1    skip git fetch/pull
#   PIC_AGENTIC_SKIP_SIM=1     install pic-agentic without the [sim] extra
#   PYTHON                     python interpreter, default python3

set -euo pipefail

HOMESERVER="${PIC_AGENTIC_HOMESERVER:-https://chat.academiccloud.de}"
ROOM_ID="${PIC_AGENTIC_ROOM_ID:-}"
RCP_SECRET="${PIC_AGENTIC_RCP_SECRET:-}"
SIM="${PIC_AGENTIC_SIM:-cluster}"
BRANCH="${PIC_AGENTIC_BRANCH:-main}"
WORKDIR="${PIC_AGENTIC_WORKDIR:-$HOME/pic-agentic}"
MESSAGE_DIR="${PIC_AGENTIC_MESSAGE_DIR:-$WORKDIR/shared}"
SIM_SETUP_ROOT="${PIC_AGENTIC_SIM_SETUP_ROOT:-$WORKDIR/sims}"
CLUSTER_TEMPLATE_DIR="${PIC_AGENTIC_CLUSTER_TEMPLATE_DIR:-}"
CLUSTER_PRESET="${PIC_AGENTIC_CLUSTER_PRESET:-}"
PICONGPU_REVISION="${PIC_AGENTIC_PICONGPU_REVISION:-122160f6125eccd0606bb8ceb11c815c88ffaf2d}"
PICONGPU_REPO_URL="${PIC_AGENTIC_PICONGPU_REPO_URL:-https://github.com/chillenzer-agents/picongpu}"
JOB_WAIT_S="${PIC_AGENTIC_JOB_WAIT_TIMEOUT_S:-600}"
ACK_S="${PIC_AGENTIC_ACK_TIMEOUT_S:-900}"
POLL_S="${PIC_AGENTIC_POLL_INTERVAL_S:-30}"
POLL_MAX_S="${PIC_AGENTIC_POLL_MAX_INTERVAL_S:-300}"
REPO_URL="https://github.com/chillenzer-agents/pic-agentic.git"
SRC="$WORKDIR/src"
VENV="$WORKDIR/venv"
CONFIG="${PIC_AGENTIC_CONFIG:-$HOME/.config/pic-agentic/config.toml}"

log() { printf '==> %s\n' "$*"; }
die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[ -n "$ROOM_ID" ] || die "set PIC_AGENTIC_ROOM_ID (the room the MCP server created)"
[ -n "$RCP_SECRET" ] || die "set PIC_AGENTIC_RCP_SECRET (must match the MCP server)"

PY="${PYTHON:-python3}"

# 1. Network preflight: the login node must reach the homeserver over HTTPS.
#    This is the one thing that cannot be verified from off-cluster.
log "checking HTTPS reachability of $HOMESERVER"
if command -v curl >/dev/null 2>&1; then
  code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$HOMESERVER/_matrix/client/versions" 2>/dev/null || true)"
  if [ "$code" != "200" ]; then
    cat >&2 <<EOF
ERROR: could not reach $HOMESERVER (HTTP ${code:-none}).
If the login node has no direct egress, set a proxy and retry:
    export https_proxy=http://<proxy>:<port>
    export http_proxy=http://<proxy>:<port>
Note: matrix-nio does not honour these automatically; if a proxy is
mandatory, stop here and report back (the transport needs a proxy knob).
EOF
    exit 1
  fi
  log "homeserver reachable (HTTP 200)"
else
  log "curl not found; skipping the reachability check"
fi

# 2. Python version check (the package needs >= 3.11).
command -v "$PY" >/dev/null 2>&1 || die "$PY not found; try 'module load python' or set PYTHON=<path>"
"$PY" - <<'PYCHECK' || die "need Python >= 3.11"
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PYCHECK
log "python: $("$PY" --version 2>&1)"

# 3. Get the source (clone once, update on re-runs).
command -v git >/dev/null 2>&1 || die "git not found; try 'module load git'"
if [ -d "$SRC/.git" ]; then
  if [ "${PIC_AGENTIC_NO_UPDATE:-0}" != "1" ]; then
    log "updating $SRC"
    git -C "$SRC" fetch --quiet origin "$BRANCH"
    git -C "$SRC" checkout --quiet "$BRANCH"
    git -C "$SRC" merge --ff-only --quiet "origin/$BRANCH" || log "not a fast-forward; staying on local $BRANCH"
  fi
else
  log "cloning $REPO_URL ($BRANCH) into $SRC"
  mkdir -p "$WORKDIR"
  git clone --quiet --branch "$BRANCH" "$REPO_URL" "$SRC"
fi

# 4. Install into a venv (idempotent).
if [ ! -x "$VENV/bin/python" ]; then
  log "creating venv at $VENV"
  "$PY" -m venv "$VENV"
fi
log "installing pic-agentic from $SRC (reuses the cache on re-runs)"
"$VENV/bin/python" -m pip install --quiet --upgrade pip
if [ "${PIC_AGENTIC_SKIP_SIM:-0}" = "1" ]; then
  log "installing without the [sim] extra (M1 hello only)"
  "$VENV/bin/python" -m pip install --quiet -e "$SRC"
else
  # The [sim] extra pulls the pinned PIConGPU from the fork (lossless Runner
  # round-trip) plus cwltool.  picongpu is installed as a git dependency of
  # the extra, so the `sdist`/`wheel` build needs git on the login node.
  log "installing with the [sim] extra (pinned picongpu + cwltool)"
  "$VENV/bin/python" -m pip install --quiet -e "${SRC}[sim]"

  # 4b. pip does NOT upgrade an already-installed git dependency when only the
  #     pinned revision changes, so the picongpu wheel silently lags the server
  #     when the pin is bumped.  The simclient then rejects every submit with
  #     `version_mismatch` (schema_hash differs) -- and the explicit
  #     PIC_AGENTIC_PICONGPU_REVISION export below masks the revision check, so
  #     the schema hash is the only signal.  Detect the drift and force-refresh.
  installed_rev="$(
    "$VENV/bin/python" - <<'PYREV'
import json
try:
    from importlib.metadata import distribution
    info = json.loads(distribution("picongpu").read_text("direct_url.json") or "{}")
    print(info.get("vcs_info", {}).get("commit_id", ""))
except Exception:
    print("")
PYREV
  )"
  if [ "$installed_rev" != "$PICONGPU_REVISION" ]; then
    log "picongpu drift: installed=${installed_rev:-none} wanted=$PICONGPU_REVISION; force-reinstalling"
    "$VENV/bin/python" -m pip install --quiet --force-reinstall --no-deps \
      "picongpu @ git+${PICONGPU_REPO_URL}@${PICONGPU_REVISION}#subdirectory=lib/python"
  else
    log "picongpu revision matches the pin ($PICONGPU_REVISION)"
  fi

  # 4c. Capability self-check: the git checkout must carry the M3 control+result
  #     wiring.  A checkout on a pre-M3 ref accepts submits but answers every
  #     result read with a hello_ack/rejected_by_policy, so the agent can never
  #     read results.  Fail loudly rather than run a client that silently cannot
  #     do the job.
  if ! grep -q "SimulationType.RESULT_COMMAND," "$SRC/src/pic_agentic/simclient/client.py"; then
    die "checkout $SRC ($BRANCH) lacks M3 result handling; set PIC_AGENTIC_BRANCH to main (the integration tip)"
  fi
  log "capability check: simclient handles control + result requests"

  # 4d. Submit-system self-check (F11).  pic-agentic no longer sends a submit
  #     command on the wire: the pinned runner takes it from
  #     ``rc_params["tbg_submit"]``, and the simclient refuses a submission
  #     unless that resolves to the one value it can follow (``sbatch``).
  #     Python ``rc_params`` does NOT read the shell ``TBG_SUBMIT`` a site
  #     profile exports -- it reads ``picongpurc.toml`` (via ``PIC_RC``, a
  #     parent dir, or ``~/.config/picongpu/``).  A clean login node therefore
  #     has it unset and every submit would be rejected with
  #     ``submit_system_mismatch``.  Verify up front and fail with the fix
  #     rather than let the agent discover it on every run.
  submit_system="$(
    "$VENV/bin/python" - <<'PYSUB'
from pic_agentic.simclient.simulation import _detect_submit_system
print(_detect_submit_system() or "")
PYSUB
  )"
  if [ "$submit_system" != "sbatch" ]; then
    die "rc_params['tbg_submit'] is ${submit_system:-unset}, but pic-agentic can only follow 'sbatch'.
       Set it for this login node, e.g.:
         printf 'tbg_submit = \"sbatch\"\\n' > ~/.config/picongpu/picongpurc.toml   # or point PIC_RC at a site toml
       (or source your site's PIConGPU profile that writes the matching picongpurc.toml)."
  fi
  log "submit-system check: rc_params['tbg_submit']=$submit_system"
fi

# 5. Interactive MAS device login (once).
if [ "${PIC_AGENTIC_SKIP_LOGIN:-0}" = "1" ] && [ -f "$CONFIG" ]; then
  log "reusing existing token config $CONFIG"
else
  log "starting the OAuth device login (one browser approval)"
  "$VENV/bin/python" "$SRC/scripts/mas_login.py" --homeserver "$HOMESERVER"
fi

# 6. Shared message directory must exist and be writable.
mkdir -p "$MESSAGE_DIR"
[ -w "$MESSAGE_DIR" ] || die "$MESSAGE_DIR is not writable"
log "message dir: $MESSAGE_DIR"

# 6b. Generated-setup root: enables the M2 submit handler when writable.
if [ "${PIC_AGENTIC_SKIP_SIM:-0}" != "1" ]; then
  mkdir -p "$SIM_SETUP_ROOT" || log "cannot create $SIM_SETUP_ROOT; M2 submit stays disabled"
  log "sim setup root: $SIM_SETUP_ROOT"
fi

# 7. Run the simclient against the real SLURM from PATH.
cat <<EOF

==> starting the simclient
    room      : $ROOM_ID
    sim       : $SIM
    slurm     : $(command -v sbatch 2>/dev/null || echo 'sbatch NOT FOUND in PATH')
    job wait  : ${JOB_WAIT_S}s   ack: ${ACK_S}s   poll: ${POLL_S}s..${POLL_MAX_S}s
    (leave this running; Ctrl-C to stop)
EOF

export PIC_AGENTIC_HOMESERVER="$HOMESERVER"
export PIC_AGENTIC_ROOM_ID="$ROOM_ID"
export PIC_AGENTIC_RCP_SECRET="$RCP_SECRET"
export PIC_AGENTIC_SIM="$SIM"
export PIC_AGENTIC_MESSAGE_DIR="$MESSAGE_DIR"
if [ "${PIC_AGENTIC_SKIP_SIM:-0}" != "1" ]; then
  export PIC_AGENTIC_SIM_SETUP_ROOT="$SIM_SETUP_ROOT"
  export PIC_AGENTIC_PICONGPU_REVISION="$PICONGPU_REVISION"
  if [ -n "$CLUSTER_TEMPLATE_DIR" ]; then
    export PIC_AGENTIC_CLUSTER_TEMPLATE_DIR="$CLUSTER_TEMPLATE_DIR"
  fi
  if [ -n "$CLUSTER_PRESET" ]; then
    export PIC_AGENTIC_CLUSTER_PRESET="$CLUSTER_PRESET"
  fi
fi
export PIC_AGENTIC_JOB_WAIT_TIMEOUT_S="$JOB_WAIT_S"
export PIC_AGENTIC_ACK_TIMEOUT_S="$ACK_S"
export PIC_AGENTIC_POLL_INTERVAL_S="$POLL_S"
export PIC_AGENTIC_POLL_MAX_INTERVAL_S="$POLL_MAX_S"
export PIC_AGENTIC_CONFIG="$CONFIG"
# Keep matrix-nio state off shared /tmp.
export PIC_AGENTIC_NIO_STORE_DIR="${PIC_AGENTIC_NIO_STORE_DIR:-$WORKDIR/nio-store}"

exec "$VENV/bin/python" -m pic_agentic.simclient
