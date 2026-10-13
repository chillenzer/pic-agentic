#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Institute of Radiation Physics, Helmholtz-Zentrum Dresden-Rossendorf
#
# SPDX-License-Identifier: MIT
#
# ============================================================================
# pic-agentic production MCP installer (clean install).
#
# Installs the pic-agentic MCP server NON-EDITABLY from a pinned, immutable git
# ref into a dedicated venv, writes the 0600 config, registers it with opencode
# and runs the preflights.  It deliberately leaves NO pic-agentic source
# checkout in the agent's reach: the server is a package in a venv, the repo,
# docs/beta-test-prompts.md, AGENTS.md and README.md are never present.
#
# Usage:
#     bash scripts/install-mcp.sh            # install + register + preflight
#     bash scripts/install-mcp.sh --check    # read-only preflight report only
#     bash scripts/install-mcp.sh reset      # clear campaign/reuse state only
#
# The default is `install`.  `--check` is the beta acceptance check: it reports
# PASS/FAIL for the venv, the config, the opencode registration, the tool
# surface and the room/secret rendezvous without changing anything.
#
# Secrets: the script never prints the RCP secret, the MAS tokens or the config
# contents.  The config file is written 0600.  Keep any local copy of this file
# 0600 if you pre-fill RCP_SECRET in the environment.
#
# All settings can be overridden via the environment (see `usage`).
# ============================================================================
set -euo pipefail

# ---- pinned revision --------------------------------------------------------
# Immutable 40-hex commit on chillenzer-agents/pic-agentic carrying the full
# M1-M4 stack (non-blocking control plane, results/plugin readers, agenda
# research loop, fleet, auth) this installer ships.  Bump deliberately; an
# override is allowed for testing but production must pin a SHA.  Pinned to the
# main merge of the wave2-beta8 integration PR #8 (the capability preflight
# below refuses a pin that lacks the required simclient features, so a stale pin
# fails loudly rather than silently installing an old client).
PIC_AGENTIC_PIN="${PIC_AGENTIC_PIN:-313d84d9e46a44d640c03e21a01e534d35eac7f8}"
PIC_AGENTIC_REPO="${PIC_AGENTIC_REPO:-https://github.com/chillenzer-agents/pic-agentic}"
RAW_REPO="${PIC_AGENTIC_RAW_REPO:-https://raw.githubusercontent.com/chillenzer-agents/pic-agentic}"

# ---- settings (override via env) -------------------------------------------
SIM="${SIM:-cluster}"
HOMESERVER="${HOMESERVER:-https://chat.academiccloud.de}"
ROOM_ID="${ROOM_ID:-!IrcSSSmxsiZrhEmwiJ:academiccloud.de}"
# Shared with the cluster-side simclient.  Required for a normal install.
RCP_SECRET="${RCP_SECRET:-}"
# Path convention only: the cluster simclient owns this dir; the server never
# writes there (the inline M2 wire goes over Matrix).
MESSAGE_DIR="${MESSAGE_DIR:-/home/lenz93/pic-agentic/shared}"
# Dedicated venv; the source is fetched only as a wheel, never checked out.
WORKDIR="${WORKDIR:-$HOME/.local/share/pic-agentic}"
VENV="${VENV:-$WORKDIR/venv}"
CONFIG="${CONFIG:-$HOME/.config/pic-agentic/config.toml}"
OPENCODE_JSON="${OPENCODE_JSON:-$HOME/.config/opencode/opencode.json}"
AGENDA_FILE="${AGENDA_FILE:-$HOME/.config/pic-agentic/campaign.json}"
PY="${PYTHON:-python3}"
# Tool-surface guard: the onboarding surface this installer is verified against.
EXPECTED_TOOLS="${PIC_AGENTIC_EXPECTED_TOOLS:-36}"
# Seconds to wait for the room backfill in the preflight.
ROOM_PREFLIGHT_TIMEOUT_S="${PIC_AGENTIC_ROOM_PREFLIGHT_TIMEOUT_S:-60}"
# MCP client request budget (ms).  Single source of truth for BOTH the opencode
# server entry `timeout` and the `PIC_AGENTIC_MCP_TIMEOUT_MS` env var the server
# reads to bound wait_for_simulation: it must be the same number, or the server
# cannot tell a wait that outlasts the client (D1).  Kept above the server's
# MAX_WAIT_TIMEOUT_S (3600 s) plus WAIT_CLIENT_TIMEOUT_SKEW_S (5 s) so even the
# longest accepted wait fits the budget and the default wait (1800 s) is never
# refused.  The default is kept equal to beta-container-setup.sh's; bump both
# together.
MCP_TIMEOUT_MS="${PIC_AGENTIC_MCP_TIMEOUT_MS:-3900000}"

MODE="install"
case "${1:-install}" in
  install | --install) MODE="install" ;;
  --check | check) MODE="check" ;;
  reset) MODE="reset" ;;
  -h | --help | help)
    sed -n '2,29p' "$0"
    exit 0
    ;;
  *)
    printf 'ERROR: unknown argument: %s (use install, --check or reset)\n' "$1" >&2
    exit 2
    ;;
esac

log() { printf '==> %s\n' "$*"; }
die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}
ok() { printf 'PASS: %s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*" >&2; }

# `reset`: clear campaign state so the next prompt starts fresh.  create_campaign
# refuses while a campaign exists and there is no delete tool.  Touches ONLY
# campaign state.
if [ "$MODE" = "reset" ]; then
  root="$(dirname "$AGENDA_FILE")"
  rm -f "$AGENDA_FILE" "$root/reuse-registry.json"
  printf 'reset: removed campaign + reuse registry under %s\n' "$root"
  exit 0
fi

# ---- preconditions ---------------------------------------------------------
command -v "$PY" >/dev/null || die "$PY not found"
"$PY" - <<'PYCHECK' || die "need Python >= 3.11"
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PYCHECK

case "$PIC_AGENTIC_PIN" in
  *[!0-9a-f]* | "") die "PIC_AGENTIC_PIN must be a lowercase 40-hex SHA, got: $PIC_AGENTIC_PIN" ;;
esac
[ "${#PIC_AGENTIC_PIN}" -eq 40 ] || die "PIC_AGENTIC_PIN must be 40 hex chars, got ${#PIC_AGENTIC_PIN}"

# A room id always starts with '!' or '#'.
if [ "$MODE" = "install" ]; then
  case "$ROOM_ID" in
    !* | \#*) ;;
    *) die "ROOM_ID resolved to ${ROOM_ID@Q}; expected '!...:server'" ;;
  esac
  # The secret is read from the prefilled 0600 config when not exported, so a
  # re-run never has to re-export it (and never echoes it).
  if [ -z "$RCP_SECRET" ] && [ -f "$CONFIG" ]; then
    RCP_SECRET="$(
      "$PY" - "$CONFIG" <<'PYSEC'
import sys
import tomllib
from pathlib import Path

try:
    print(tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("pic_agentic", {}).get("rcp_secret", ""))
except (OSError, ValueError):
    print("")
PYSEC
    )"
  fi
  [ -n "$RCP_SECRET" ] || die "export RCP_SECRET=<64 hex chars; see the cluster side>"
fi

# ---- config dirs (root-owned ~/.config is fixed with sudo) -----------------
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
    if ! sudo mkdir -p "$dir" || ! sudo chown "$(id -u):$(id -g)" "$dir"; then
      die "cannot create $dir"
    fi
  fi
}

# ============================================================================
# Preflight helpers (used by both `install` and `--check`).
# ============================================================================

# Tool-surface preflight: build the server and assert the expected tool count
# and that the seeded `instructions` are present.  Needs no network.
preflight_tools() {
  PIC_AGENTIC_CONFIG="$CONFIG" "$VENV/bin/python" - "$EXPECTED_TOOLS" <<'PYFLY'
import asyncio
import sys

from pic_agentic.config import Config
from pic_agentic.server.app import build_server

expected = int(sys.argv[1])
c = Config.load()
missing = [
    n
    for n in ("homeserver", "user_id", "access_token", "room_id", "rcp_secret", "message_dir")
    if not getattr(c, n)
]
if missing:
    sys.exit(f"config incomplete: {missing} (check the config path)")
server, _ = build_server(c, "cluster")


async def main() -> None:
    names = {t.name for t in await server.list_tools()}
    for want in ("hello", "submit_simulation", "build_spec", "create_campaign", "advance_agenda", "run_analysis"):
        if want not in names:
            sys.exit(f"missing tool: {want}")
    if len(names) != expected:
        sys.exit(f"tool count mismatch: got {len(names)}, expected {expected}")
    if not server.instructions:
        sys.exit("instructions are not seeded")
    print(f"OK - {len(names)} tools; instructions seeded: {bool(server.instructions)}")


asyncio.run(main())
PYFLY
}

# Installed-revision + capability preflight: the venv's pic-agentic must be the
# pinned commit and its simclient must carry the M3 result handling and the
# non-blocking control plane this installer ships.  This is the check that
# makes a stale/mismatched PIC_AGENTIC_PIN fail loudly instead of silently
# installing the old blocking client (the live beta timeouts).  Mirrors
# cluster_simclient.sh's capability self-check.
preflight_installed() {
  "$VENV/bin/python" - "$PIC_AGENTIC_PIN" <<'PYREV'
import json
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

pin = sys.argv[1]
try:
    dist = distribution("pic-agentic")
except PackageNotFoundError:
    sys.exit("pic-agentic is not installed in the venv")
direct_url = dist.read_text("direct_url.json") or ""
installed = ""
try:
    installed = json.loads(direct_url).get("vcs_info", {}).get("commit_id", "")
except ValueError:
    installed = ""
if not installed:
    sys.exit("pic-agentic has no recorded VCS revision (not a git install?)")
if installed != pin:
    sys.exit(f"installed revision {installed} != pin {pin} (stale install)")

import pic_agentic.simclient.client as client_mod
import pic_agentic.simclient.simulation as sim_mod

client_source = Path(client_mod.__file__).read_text(encoding="utf-8")
source = client_source + Path(sim_mod.__file__).read_text(encoding="utf-8")
required = {
    "M3 result handling": "SimulationType.RESULT_COMMAND",
    "non-blocking dispatch": "_dispatch_message",
    "build gate": "_build_semaphore",
    "per-run stderr capture": "_CapturingStderr",
}
missing = [label for label, marker in required.items() if marker not in source]
if missing:
    sys.exit(f"installed simclient lacks: {', '.join(missing)}")
print(f"OK - installed revision {installed}; simclient carries M3 + non-blocking dispatch")
PYREV
}

# Room/secret rendezvous preflight: connect to the room, backfill the recent
# timeline and require at least one RCP message (from either role -- a fresh
# host has no server-role traffic yet) to verify under the configured secret.
# This catches the live bad-secret failure (config secret != the secret the
# cluster signs with) before declaring success, while not failing a correct
# fresh install just because no MCP tool has run yet.
preflight_room() {
  PIC_AGENTIC_CONFIG="$CONFIG" "$VENV/bin/python" - "$ROOM_PREFLIGHT_TIMEOUT_S" <<'PYROOM'
import asyncio
import sys

from pic_agentic.auth import MasTokenStore
from pic_agentic.config import Config
from pic_agentic.transport.matrix import MatrixTransport

timeout = float(sys.argv[1])
BAD_SECRET = (
    "secret does not match the room; re-mint with `fresh` or update the secret "
    "(the configured rcp_secret verifies none of the room's RCP messages)"
)


async def main() -> int:
    c = Config.load()
    c.require("homeserver", "user_id", "access_token", "room_id", "rcp_secret")
    token_provider = MasTokenStore.from_config(c).access_token if c.has_refresh_chain() else None
    transport = MatrixTransport(
        c.homeserver,
        c.user_id,
        c.access_token,
        c.room_id,
        token_provider=token_provider,
    )
    try:
        messages = await asyncio.wait_for(transport.backfill(), timeout=timeout)
    except TimeoutError:
        sys.exit(f"room preflight timed out after {timeout:.0f}s (homeserver unreachable?)")
    finally:
        await transport.close()
    if not messages:
        # A fresh host: the server has not run yet and the cluster may not have
        # posted either.  This is NOT a bad secret; warn and pass so a correct
        # first install is not aborted.
        print(
            "WARN - no RCP messages in the room yet; secret consistency is unverified "
            "(re-run --check after the first exchange)",
        )
        return 0
    # Both roles sign with the same shared secret, so whichever role has posted
    # is valid evidence.  A fresh host has only simclient-role messages (the
    # cluster side), never server-role ones, so verifying server-only was a
    # false negative.
    verified = sum(1 for m in messages if m.verify(c.rcp_secret))
    if verified == 0:
        print(BAD_SECRET, file=sys.stderr)
        print(f"(0 of {len(messages)} RCP messages verified)", file=sys.stderr)
        return 1
    roles = sorted({m.sender_role.value for m in messages if m.verify(c.rcp_secret)})
    print(f"OK - {verified}/{len(messages)} RCP messages verify under the configured secret (roles: {', '.join(roles)})")
    return 0


raise SystemExit(asyncio.run(main()))
PYROOM
}

# Registration check: the opencode.json entry exists and is shaped correctly.
preflight_registration() {
  "$PY" - "$OPENCODE_JSON" "$VENV" "$CONFIG" <<'PYREG'
import json
import sys
from pathlib import Path

path, venv, config = sys.argv[1], sys.argv[2], sys.argv[3]
p = Path(path)
if not p.exists():
    sys.exit(f"no opencode config at {path}")
cfg = json.loads(p.read_text(encoding="utf-8"))
entry = (cfg.get("mcp") or {}).get("pic-agentic")
if not entry:
    sys.exit("no 'pic-agentic' MCP server registered")
problems = []
if entry.get("type") != "local":
    problems.append("type != local")
if entry.get("command") != [f"{venv}/bin/pic-agentic-mcp"]:
    problems.append("command does not point at the venv entry point")
if not entry.get("enabled"):
    problems.append("not enabled")
env = entry.get("environment") or {}
if env.get("PIC_AGENTIC_SIM") != "cluster":
    problems.append("PIC_AGENTIC_SIM != cluster")
if env.get("PIC_AGENTIC_CONFIG") != config:
    problems.append("PIC_AGENTIC_CONFIG does not match")
timeout = entry.get("timeout")
if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
    problems.append("timeout missing/invalid")
# The server bounds wait_for_simulation against PIC_AGENTIC_MCP_TIMEOUT_MS, so
# it must carry the SAME budget as the opencode entry timeout (D1).
env_budget = env.get("PIC_AGENTIC_MCP_TIMEOUT_MS")
if env_budget != str(entry.get("timeout")):
    problems.append("PIC_AGENTIC_MCP_TIMEOUT_MS != timeout")
if problems:
    sys.exit("; ".join(problems))
print(f"OK - registered (timeout {entry['timeout']} ms)")
PYREG
}

# ============================================================================
# `--check`: read-only report.  Runs every preflight, never writes.
# ============================================================================
if [ "$MODE" = "check" ]; then
  failed=0
  log "check: venv + config"
  if [ -x "$VENV/bin/pic-agentic-mcp" ]; then
    ok "venv entry point: $VENV/bin/pic-agentic-mcp"
  else
    bad "venv entry point missing: $VENV/bin/pic-agentic-mcp (run the installer)"
    failed=1
  fi
  if [ -f "$CONFIG" ]; then
    perms="$(stat -c '%a' "$CONFIG")"
    if [ "$perms" = "600" ]; then
      ok "config $CONFIG (mode 0600)"
    else
      bad "config $CONFIG has mode $perms (expected 600)"
      failed=1
    fi
  else
    bad "config missing: $CONFIG"
    failed=1
  fi

  if [ -x "$VENV/bin/python" ] && [ -f "$CONFIG" ]; then
    log "check: installed revision + simclient capability"
    if preflight_installed; then ok "installed revision/capability"; else
      bad "installed revision/capability"
      failed=1
    fi
    log "check: MCP registration"
    if preflight_registration; then ok "opencode registration"; else
      bad "opencode registration"
      failed=1
    fi
    log "check: tool surface"
    if preflight_tools; then ok "tool surface"; else
      bad "tool surface"
      failed=1
    fi
    log "check: room/secret rendezvous"
    if preflight_room; then ok "room/secret rendezvous"; else
      bad "room/secret rendezvous"
      failed=1
    fi
  else
    bad "skipping registration/tool/room checks (venv or config missing)"
    failed=1
  fi

  if [ "$failed" -ne 0 ]; then
    printf '\ncheck FAILED\n' >&2
    exit 1
  fi
  printf '\ncheck PASSED\n'
  exit 0
fi

# ============================================================================
# `install`
# ============================================================================
log "settings: room=$ROOM_ID sim=$SIM"
log "          pin=$PIC_AGENTIC_PIN"
log "          venv=$VENV"
log "          config=$CONFIG"
log "          msgdir=$MESSAGE_DIR"

ensure_dir "$(dirname "$CONFIG")"
ensure_dir "$(dirname "$OPENCODE_JSON")"
ensure_dir "$WORKDIR"

# 1. Non-editable install from the pinned ref into the dedicated venv.  pip
#    resolves the `[sim]` extra, which pulls the pinned picongpu too.  Nothing
#    is cloned into the workspace and there is no -e.
[ -x "$VENV/bin/python" ] || {
  log "creating venv at $VENV"
  "$PY" -m venv "$VENV"
}
log "installing pic-agentic[sim] @ $PIC_AGENTIC_PIN (non-editable)"
log "  the [sim] extra fetches + builds the pinned picongpu wheel (5-15 min cold)"
if [ -d /workspace/.beta-pip-cache ]; then
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/.beta-pip-cache}"
  log "  using shared pip cache: $PIP_CACHE_DIR"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
# A normal (not --no-deps) install: the `[sim]` extra's pinned picongpu and
# every runtime dependency must land in the venv.
"$VENV/bin/python" -m pip install --quiet \
  "pic-agentic[sim] @ git+${PIC_AGENTIC_REPO}@${PIC_AGENTIC_PIN}"
[ -x "$VENV/bin/pic-agentic-mcp" ] || die "install did not produce $VENV/bin/pic-agentic-mcp"
# Refuse a stale pin here (not just in --check): the installed revision and
# simclient capabilities must match what we advertise, so the old blocking
# control plane can never be shipped silently.
log "asserting the installed revision + simclient capability"
preflight_installed || die "installed pic-agentic does not match the pin / lacks the M3 control plane"

# 2. One-time Matrix device login writes the token fields and preserves any keys
#    already present.  scripts/mas_login.py is not part of the installed wheel,
#    so it is fetched from the SAME pinned ref, run, and deleted; no checkout is
#    left behind.  An existing config reuses its login unless forced.
mkdir -p "$(dirname "$CONFIG")"
if [ -f "$CONFIG" ] && [ "${PIC_AGENTIC_FORCE_LOGIN:-0}" != "1" ]; then
  log "reusing existing login in $CONFIG (PIC_AGENTIC_FORCE_LOGIN=1 to redo)"
else
  login_script="${PIC_AGENTIC_LOGIN_SCRIPT:-}"
  cleanup_login=0
  if [ -z "$login_script" ]; then
    login_script="$(mktemp)"
    cleanup_login=1
    log "fetching mas_login.py from the pinned ref"
    curl -fsSL "${RAW_REPO}/${PIC_AGENTIC_PIN}/scripts/mas_login.py" -o "$login_script" ||
      die "could not fetch mas_login.py (set PIC_AGENTIC_LOGIN_SCRIPT to a local copy)"
  fi
  log "starting the OAuth device login (one browser approval)"
  PIC_AGENTIC_CONFIG="$CONFIG" "$VENV/bin/python" "$login_script" --homeserver "$HOMESERVER" --out "$CONFIG" ||
    {
      [ "$cleanup_login" = "1" ] && rm -f "$login_script"
      die "device login failed"
    }
  [ "$cleanup_login" = "1" ] && rm -f "$login_script"
fi

# 3. Add the room / shared secret / paths the login script does not manage.
#    Values are passed as argv, never interpolated into a shell command.
log "adding room_id / rcp_secret / message_dir / agenda_file to $CONFIG"
PIC_AGENTIC_CONFIG="$CONFIG" "$VENV/bin/python" - "$CONFIG" "$ROOM_ID" "$RCP_SECRET" "$MESSAGE_DIR" "$AGENDA_FILE" <<'PYCFG'
import sys
from pathlib import Path

path, room, secret, msgdir, agenda = sys.argv[1:6]
if not room or not secret:
    sys.exit(f"refusing to write an empty room/secret (room={room!r}, secret_len={len(secret)})")
p = Path(path)
text = p.read_text(encoding="utf-8") if p.exists() else "[pic_agentic]\n"
lines = [
    ln
    for ln in text.splitlines()
    if not ln.startswith(("room_id", "rcp_secret", "message_dir", "agenda_file"))
]
lines += [
    f'room_id = "{room}"',
    f'rcp_secret = "{secret}"',
    f'message_dir = "{msgdir}"',
    f'agenda_file = "{agenda}"',
]
p.write_text("\n".join(lines) + "\n", encoding="utf-8")
PYCFG
chmod 600 "$CONFIG"

# 4. Register the MCP server with opencode (idempotent JSON edit, valid JSON
#    out, other entries preserved).  NOTE: opencode's key is `environment`, not
#    `env`.  JSON cannot carry comments, so the rationale lives here: `timeout`
#    is the request budget (default 3900000 ms) and the SAME value is stamped
#    into `PIC_AGENTIC_MCP_TIMEOUT_MS` so the server can bound
#    wait_for_simulation against it (D1).
log "registering the pic-agentic MCP server in $OPENCODE_JSON"
"$VENV/bin/python" - "$OPENCODE_JSON" "$VENV" "$CONFIG" "$MCP_TIMEOUT_MS" <<'PYJSON'
import json
import sys
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
        "PIC_AGENTIC_MCP_TIMEOUT_MS": str(timeout_ms),
    },
    "timeout": int(timeout_ms),
}
p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
PYJSON
chmod 600 "$OPENCODE_JSON" 2>/dev/null || true

# 5. Preflights.  A failure aborts (set -e), so "ready" is never printed on a
#    config the server would reject.
log "preflight: tool surface"
preflight_tools || die "tool preflight failed"
log "preflight: room/secret rendezvous"
preflight_room || die "room/secret preflight failed"

cat <<EOF

==> Agent container ready.
    venv   : $VENV
    config : $CONFIG
    room   : $ROOM_ID   sim: $SIM
    campaign: $AGENDA_FILE

NEXT:
  1. RESTART the opencode session so it loads the 'pic-agentic' MCP server.
  2. On the CLUSTER login node, start the simclient with THIS room + secret and
     the SAME pinned stack (main is now the M1-M4 integration tip):
       PIC_AGENTIC_BRANCH='$PIC_AGENTIC_PIN' \\
       PIC_AGENTIC_ROOM_ID='$ROOM_ID' \\
       PIC_AGENTIC_RCP_SECRET='<the same secret; do not paste it here>' \\
       PIC_AGENTIC_MESSAGE_DIR='$MESSAGE_DIR' \\
       bash cluster_simclient.sh
  3. Reset campaign state between prompts (no reinstall, no re-login):
       bash scripts/install-mcp.sh reset
  4. Re-run the acceptance checks any time (read-only):
       bash scripts/install-mcp.sh --check
EOF
