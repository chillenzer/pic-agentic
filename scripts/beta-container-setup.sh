#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT
#
# ============================================================================
# pic-agentic BETA-TEST bootstrap - run in an EMPTY, isolated container that has
# only git + python3.11+ + opencode.
#
# It clones (or reuses) the pic-agentic beta ref, installs the MCP server, does
# the one-time Matrix device login, mints a fresh room + RCP secret, writes the
# config, registers the MCP server with opencode and preflights it.
#
# Run it interactively (the device login needs a browser):
#     git clone https://github.com/chillenzer-agents/pic-agentic ~/pic-agentic-beta
#     bash ~/pic-agentic-beta/scripts/beta-container-setup.sh fresh
#
# Then RESTART the opencode session so it picks up the new MCP server.
# Keep this file 0600 - it can contain the shared RCP secret. Never commit it.
#
# BETWEEN BETA PROMPTS reset the campaign state (no reinstalls, no re-login):
#     bash scripts/beta-container-setup.sh reset
# ============================================================================
set -euo pipefail

# ---- modes that do not require an RCP secret -------------------------------
# reset: clear the campaign so the next prompt starts fresh.  create_campaign
# refuses if any campaign exists and there is no delete tool, so a new agenda
# needs its files gone.  Touches ONLY campaign state.
if [ "${1:-}" = "reset" ]; then
  root="$(dirname "${AGENDA_FILE:-$HOME/.config/pic-agentic/campaign.json}")"
  rm -f "${AGENDA_FILE:-$HOME/.config/pic-agentic/campaign.json}" \
    "$root/reuse-registry.json"
  printf 'reset: removed campaign + reuse registry under %s\n' "$root"
  exit 0
fi

# ---- settings (override via env if needed) --------------------------------
ROOM_ID="${ROOM_ID:-!IrcSSSmxsiZrhEmwiJ:academiccloud.de}"
SIM="${SIM:-cluster}"
HOMESERVER="${HOMESERVER:-https://chat.academiccloud.de}"
# Shared with the cluster-side simclient (cluster_snippets.sh).  Required for a
# normal (re)install; the `fresh` mode mints a new one instead.
RCP_SECRET="${RCP_SECRET:-}"
# Path convention only: the cluster simclient owns this dir; the server never
# writes there (the inline M2 wire goes over Matrix).
MESSAGE_DIR="${MESSAGE_DIR:-/home/lenz93/pic-agentic/shared}"
BRANCH="${BRANCH:-main}"
REPO_URL="${REPO_URL:-https://github.com/chillenzer-agents/pic-agentic.git}"
# Venv/cache live OUTSIDE any checkout so a clone-based run does not litter the
# repo.  Override WORKDIR to relocate.
WORKDIR="${WORKDIR:-$HOME/.local/share/pic-agentic}"
CONFIG="${CONFIG:-$HOME/.config/pic-agentic/config.toml}"
OPENCODE_JSON="${OPENCODE_JSON:-$HOME/.config/opencode/opencode.json}"
# Where the agent's campaign state lives (server-writable, container-local).
AGENDA_FILE="${AGENDA_FILE:-$HOME/.config/pic-agentic/campaign.json}"
# MCP client request budget (ms).  Single source of truth for BOTH the opencode
# server entry `timeout` and the `PIC_AGENTIC_MCP_TIMEOUT_MS` env var the server
# reads to bound wait_for_simulation: it must be the same number, or the server
# cannot tell a wait that outlasts the client (D1).  Kept above the server's
# MAX_WAIT_TIMEOUT_S (3600 s) plus WAIT_CLIENT_TIMEOUT_SKEW_S (5 s) so even the
# longest accepted wait fits the budget and the default wait (1800 s) is never
# refused.  Kept equal to install-mcp.sh's default; bump both together.
MCP_TIMEOUT_MS="${PIC_AGENTIC_MCP_TIMEOUT_MS:-3900000}"

log() { printf '==> %s\n' "$*"; }
die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

SRC="$WORKDIR/src"
VENV="$WORKDIR/venv"
PY="${PYTHON:-python3}"

# `fresh` mints a NEW room + secret after the install below, for a clean beta
# run (the server replays the room history into its sim registry at startup and
# the engine reads the persisted campaign, so reusing a room across sessions
# taints the new agent).  It is a flag that flows through the normal install
# path, not a separate early branch, so it works on a brand-new container too.
FRESH=0
[ "${1:-}" = "fresh" ] && FRESH=1

# Everything below is the normal (re)install path.  A shared secret is required
# unless we are minting a fresh room+secret.
if [ "$FRESH" = "1" ]; then
  ROOM_ID=""
else
  [ -n "$RCP_SECRET" ] || die "export RCP_SECRET=<64 hex chars; see cluster_snippets.sh> (or use '$0 fresh')"
fi

# Fail loudly on an empty room rather than writing room_id = "" and limping on
# (the beta bug that produced "config incomplete: ['room_id']" while reporting
# success). A room id always starts with '!' or '#'.
case "$ROOM_ID" in
  !* | \#*) ;;
  *)
    [ "$FRESH" = "1" ] || die "ROOM_ID resolved to ${ROOM_ID@Q}; expected '!...:server' (set ROOM_ID explicitly)"
    ;;
esac

log "settings: room=${ROOM_ID:-<fresh>} sim=$SIM branch=$BRANCH fresh=$FRESH"
log "          config=$CONFIG"
log "          secret_len=${#RCP_SECRET} msgdir=$MESSAGE_DIR"

command -v "$PY" >/dev/null || die "$PY not found"
"$PY" - <<'PYCHECK' || die "need Python >= 3.11"
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PYCHECK

# 0. Make sure the config dirs exist and are ours. Some containers ship a
#    root-owned ~/.config, in which case we create our own subdirs (and fix
#    ownership of ours) with sudo when available.
ensure_dir() {
  local dir="$1"
  if [ -d "$dir" ]; then
    [ -w "$dir" ] && return 0
    log "fixing ownership of $dir (needs sudo)"
    sudo chown -R "$(id -u):$(id -g)" "$dir" 2>/dev/null ||
      die "$dir is not writable and could not be chowned; re-run with sudo or set CONFIG/OPENCODE_JSON elsewhere"
    return 0
  fi
  local parent
  parent="$(dirname "$dir")"
  if [ -w "$parent" ]; then
    mkdir -p "$dir"
  else
    log "$parent not writable; creating $dir with sudo"
    if ! { sudo mkdir -p "$dir" && sudo chown "$(id -u):$(id -g)" "$dir"; }; then
      die "cannot create $dir"
    fi
  fi
}
ensure_dir "$(dirname "$CONFIG")"
ensure_dir "$(dirname "$OPENCODE_JSON")"
ensure_dir "$WORKDIR"

# 1. Resolve the source tree.  If this script runs from inside a pic-agentic
#    checkout (the normal empty-container route: clone the branch, then run
#    scripts/beta-container-setup.sh), use THAT tree; otherwise clone the branch.
#    Code visibility is accepted for the beta, so a checkout is fine.
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$_script_dir/../pyproject.toml" ] && grep -q "name = \"pic-agentic\"" "$_script_dir/../pyproject.toml" 2>/dev/null; then
  SRC="$(cd "$_script_dir/.." && pwd)"
  log "using the checkout this script lives in: $SRC ($(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo no-git))"
else
  if [ -d "$SRC/.git" ]; then
    log "updating $SRC"
    git -C "$SRC" fetch --quiet origin "$BRANCH"
    git -C "$SRC" checkout --quiet "origin/$BRANCH"
  else
    log "cloning $BRANCH into $SRC"
    mkdir -p "$WORKDIR"
    git clone --quiet --branch "$BRANCH" "$REPO_URL" "$SRC"
  fi
  log "checkout: $(git -C "$SRC" log --oneline -1)"
fi

# 2. venv + [sim] extra (pins picongpu from the fork; builds a wheel - minutes).
[ -x "$VENV/bin/python" ] || {
  log "creating venv at $VENV"
  "$PY" -m venv "$VENV"
}
log "installing pic-agentic with the [sim] extra (fetches + builds the pinned picongpu wheel)"
log "  this clones picongpu (with submodules) and builds a ~9 MB wheel; on a cold"
log "  container it commonly takes 5-15 min.  Progress follows (PIC_AGENTIC_VERBOSE=0 to hide):"
# A pip cache on the shared mount survives container rebuilds, so repeated beta
# containers reuse the downloaded wheels instead of re-fetching ~100 packages.
if [ -d /workspace/.beta-pip-cache ]; then
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.beta-pip-cache}"
  log "  using shared pip cache: $PIP_CACHE_DIR"
fi
if [ "${PIC_AGENTIC_VERBOSE:-1}" = "1" ]; then
  "$VENV/bin/python" -m pip install --upgrade pip
  "$VENV/bin/python" -m pip install -e "${SRC}[sim]"
else
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet -e "${SRC}[sim]"
fi

# 2b. pip does not upgrade an already-installed git dependency when only the
#     pinned revision changes; left alone, the beta container's picongpu wheel
#     lags the server and every submit is rejected with `version_mismatch`
#     (schema_hash differs).  Detect and force-refresh, mirroring the cluster
#     script.
PINNED_REV="$(grep -oP 'picongpu@\K[0-9a-f]{40}' "$SRC/pyproject.toml" | head -1)"
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
if [ -n "$PINNED_REV" ] && [ "$installed_rev" != "$PINNED_REV" ]; then
  log "picongpu drift: installed=${installed_rev:-none} pinned=$PINNED_REV; force-reinstalling"
  "$VENV/bin/python" -m pip install --quiet --force-reinstall --no-deps \
    "picongpu @ git+https://github.com/chillenzer-agents/picongpu@${PINNED_REV}#subdirectory=lib/python"
else
  log "picongpu revision matches the pin (${PINNED_REV:-unknown})"
fi

# 3. One-time Matrix device login; writes client_id/tokens into $CONFIG and
#    preserves any keys already present there.  Re-runs with an existing config
#    skip the browser approval unless PIC_AGENTIC_FORCE_LOGIN=1.
#    The device-login helper is not shipped in the wheel, so fetch it from the
#    pinned ref (the checkout is present either way; code visibility accepted).
mkdir -p "$(dirname "$CONFIG")"
if [ -f "$CONFIG" ] && [ "${PIC_AGENTIC_FORCE_LOGIN:-0}" != "1" ]; then
  log "reusing existing login in $CONFIG (PIC_AGENTIC_FORCE_LOGIN=1 to redo)"
else
  log "starting the OAuth device login (one browser approval)"
  "$VENV/bin/python" "$SRC/scripts/mas_login.py" --homeserver "$HOMESERVER" --out "$CONFIG"
fi

# 3b. `fresh`: mint a brand-new room + secret for an untainted run (the server
#     replays room history into its sim registry, so a reused room leaks the
#     previous run's sims to the new agent).  Use the SHIPPED entry point
#     (pic-agentic-setup), which wraps the same room creation.
if [ "$FRESH" = "1" ]; then
  log "minting a fresh room + RCP secret"
  fresh_state="$CONFIG.fresh-state.json"
  PIC_AGENTIC_CONFIG="$CONFIG" "$VENV/bin/pic-agentic-setup" \
    --state "$fresh_state" --sim "$SIM" --message-dir "$MESSAGE_DIR" >/dev/null
  ROOM_ID="$("$PY" -c "import json;print(json.load(open('$fresh_state'))['room_id'])")"
  RCP_SECRET="$("$PY" -c "import json;print(json.load(open('$fresh_state'))['rcp_secret'])")"
  log "fresh room: $ROOM_ID"
fi

# 4. Add the room / shared secret / paths the login script does not manage.
#    Env overrides win at runtime, but persist here so the MCP server is
#    self-contained. Python (tomllib/tomli absent in 3.10; use text append).
log "adding room_id / rcp_secret / message_dir / agenda_file to $CONFIG"
"$PY" - "$CONFIG" "$ROOM_ID" "$RCP_SECRET" "$MESSAGE_DIR" "$AGENDA_FILE" <<'PYCFG'
import sys
from pathlib import Path
path, room, secret, msgdir, agenda = sys.argv[1:6]
if not room or not secret:
    sys.exit(f"ERROR: refusing to write an empty room/secret (room={room!r}, secret_len={len(secret)})")
p = Path(path)
text = p.read_text(encoding="utf-8") if p.exists() else "[pic_agentic]\n"
lines = [ln for ln in text.splitlines()
         if not ln.startswith(("room_id", "rcp_secret", "message_dir", "agenda_file"))]
lines += [f'room_id = "{room}"', f'rcp_secret = "{secret}"',
          f'message_dir = "{msgdir}"', f'agenda_file = "{agenda}"']
p.write_text("\n".join(lines) + "\n", encoding="utf-8")
PYCFG
chmod 600 "$CONFIG"

# 4b. A fresh run starts from an empty slate: clear any leftover campaign and
#     reuse registry (create_campaign refuses while a campaign file exists).
if [ "$FRESH" = "1" ]; then
  rm -f "$AGENDA_FILE" "$(dirname "$AGENDA_FILE")/reuse-registry.json"
  log "cleared campaign + reuse registry under $(dirname "$AGENDA_FILE")"
fi

# 5. Register the MCP server with opencode (idempotent JSON edit).
log "registering the pic-agentic MCP server in $OPENCODE_JSON"
mkdir -p "$(dirname "$OPENCODE_JSON")"
"$PY" - "$OPENCODE_JSON" "$VENV" "$CONFIG" "$MCP_TIMEOUT_MS" <<'PYJSON'
import json, sys
from pathlib import Path
path, venv, config, timeout_ms = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
p = Path(path)
cfg = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"$schema": "https://opencode.ai/config.json"}
cfg.setdefault("mcp", {})["pic-agentic"] = {
    "type": "local",
    "command": [f"{venv}/bin/pic-agentic-mcp"],
    "enabled": True,
    "environment": {
        "PIC_AGENTIC_SIM": "cluster",
        "PIC_AGENTIC_CONFIG": config,
        # Same budget the entry `timeout` uses, so the server can refuse a
        # wait_for_simulation that would outlast the client (D1).
        "PIC_AGENTIC_MCP_TIMEOUT_MS": str(timeout_ms),
    },
    # opencode defaults MCP requests to 5000 ms; result reads cross to the
    # cluster over Matrix and can take longer, so raise the client budget.
    "timeout": int(timeout_ms),
}
p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PYJSON

# 6. Preflight: the server must start and expose the onboarding tools.  A
#    failure here aborts the script (set -e), so "ready" is never printed on a
#    config the server would reject.
log "preflight: starting the MCP server briefly"
PIC_AGENTIC_CONFIG="$CONFIG" "$VENV/bin/python" - <<'PYFLY'
import asyncio, sys
from pic_agentic.config import Config
from pic_agentic.server.app import build_server
c = Config.load()
missing = [n for n in ("homeserver", "user_id", "access_token", "room_id", "rcp_secret", "message_dir") if not getattr(c, n)]
if missing:
    sys.exit(f"ERROR: config incomplete: {missing} (check $CONFIG)")
s, _ = build_server(c, "cluster")
async def main():
    names = {t.name for t in await s.list_tools()}
    for want in ("hello", "submit_simulation", "build_spec", "create_campaign", "advance_agenda", "run_analysis"):
        if want not in names:
            sys.exit(f"ERROR: missing tool: {want}")
    print(f"OK - {len(names)} tools; instructions seeded: {bool(s.instructions)}")
asyncio.run(main())
PYFLY

cat <<EOF

==> Beta container ready.$([ "$FRESH" = "1" ] && echo "  (FRESH room minted)")
    source : $SRC  ($(git -C "$SRC" rev-parse --short HEAD))
    venv   : $VENV
    config : $CONFIG
    room   : $ROOM_ID   sim: $SIM
    campaign: $AGENDA_FILE

NEXT:
  1. RESTART the opencode session so it loads the 'pic-agentic' MCP server.
  2. Start the cluster-side simclient on the SAME pinned ref + room + secret
     (on the login node; see cluster_snippets.sh):
       PIC_AGENTIC_BRANCH='main' \\
       PIC_AGENTIC_ROOM_ID='$ROOM_ID' \\
       PIC_AGENTIC_RCP_SECRET='$RCP_SECRET' bash cluster_simclient.sh
  3. Paste one of the beta prompts from docs/beta-test-prompts.md.
EOF
