#!/usr/bin/env bash
# build_evalchemy_gpu_kaniko.sh — in-cluster kaniko build of the :evalchemy-gpu image.
#
# Mirrors OpenThoughts-Agent's docker/build_tpu_kaniko.sh (see the build-tpu-image-iris
# skill). Runs INSIDE an iris job whose task-image is docker.io/library/ubuntu:22.04
# (kaniko's executor is distroless / no bash, so it cannot be the task image directly);
# we crane-export the kaniko executor rootfs over / and run /kaniko/executor.
#
# The Dockerfile CLONES the evalchemy fork at EVALCHEMY_COMMIT (mirrors :evalchemy-tpu),
# so the kaniko build context is tiny -- just this Dockerfile. The iris workspace bundle
# therefore does NOT need to carry the 50 MB evalchemy repo (which exceeds iris's 25 MB
# bundle cap); launch from a minimal scratch dir holding only infra/docker/. The Dockerfile
# installs the reconciled deps with `uv sync --frozen` (dependency-ground-truth-uv: the
# committed uv.lock is the sole source of truth; no runtime --with/--constraint/pip hacks).
#
# Required env (passed by the iris launch as -e):
#   DOCKER_USER_ID    ghcr user (penfever)
#   DOCKER_TOKEN      a GitHub PAT with write:packages (from `gh auth token`; NOT the
#                     Docker Hub dckr_pat_ in secrets.env)
#   GITSHA            evalchemy short commit sha for the immutable :evalchemy-gpu-<gitsha> tag
#   EVALCHEMY_COMMIT  (optional) full sha/ref to clone+bake; defaults to GITSHA
#
# PINNED-FIRST: this pushes ONLY the immutable :evalchemy-gpu-<gitsha> tag. Promote the
# floating :evalchemy-gpu (crane tag) separately, AFTER a live eval smoke produces a real
# score, so a botched build cannot break other users' evalchemy-gpu jobs.
# Do not enable xtrace until after the GHCR credential has been consumed below:
# shell expansion of the required-env checks would otherwise copy DOCKER_TOKEN
# into the durable Iris task log.
set -euo pipefail

: "${DOCKER_USER_ID:?}"
: "${DOCKER_TOKEN:?}"
: "${GITSHA:?}"

# SINGLE_SNAPSHOT=0 (per-instruction layers) is the DEFAULT: it dodges the un-pullable
# ~16 GB single-blob layer (containerd restarts the single-blob GET from 0 over the ghcr
# egress and dies), giving small independently-retriable layers. The lean image is small
# anyway; keep 0. (If you build WITH `--extra vllm` you MUST keep 0 and additionally split
# the torch/nvidia install so no single layer exceeds ~8 GB -- see coreweave_gpu_ops.md.)
SINGLE_SNAPSHOT="${SINGLE_SNAPSHOT:-0}"
if [ "$SINGLE_SNAPSHOT" = "1" ]; then SNAPSHOT_FLAG="--single-snapshot"; else SNAPSHOT_FLAG=""; fi

DOCKERFILE="${DOCKERFILE:-infra/docker/Dockerfile.evalchemy-gpu}"
EVALCHEMY_COMMIT="${EVALCHEMY_COMMIT:-${GITSHA}}"
CACHE_REPO=ghcr.io/open-thoughts/openthoughts-agent/cache-evalchemy-gpu
DEST_PINNED=ghcr.io/open-thoughts/openthoughts-agent:evalchemy-gpu-${GITSHA}

# --- 1. fetch crane (static binary) ---
apt-get update -y && apt-get install -y --no-install-recommends ca-certificates curl tar
cd /tmp
CRANE_VER=v0.20.2
curl -fsSL "https://github.com/google/go-containerregistry/releases/download/${CRANE_VER}/go-containerregistry_Linux_x86_64.tar.gz" -o crane.tgz
tar -xzf crane.tgz crane
install -m 0755 crane /usr/local/bin/crane

# --- 2. crane-export the kaniko executor rootfs over / ---
crane export gcr.io/kaniko-project/executor:latest - | tar -xf - -C / || true

# --- 3. write the ghcr auth config AFTER the overlay (kaniko clobbers /kaniko otherwise) ---
export DOCKER_CONFIG=/kaniko/.docker
mkdir -p "$DOCKER_CONFIG"
# Disable `set -x` around the secret so the base64 ghcr PAT is not echoed into the job log.
{ set +x; } 2>/dev/null
AUTH=$(printf '%s:%s' "$DOCKER_USER_ID" "$DOCKER_TOKEN" | base64 | tr -d '\n')
cat > "$DOCKER_CONFIG/config.json" <<EOF
{"auths":{"ghcr.io":{"auth":"${AUTH}"}}}
EOF
unset AUTH
set -x

# --- 4. run kaniko (pinned tag ONLY; floating :evalchemy-gpu promoted after a live smoke) ---
exec /kaniko/executor \
  --context dir:///app \
  --dockerfile "${DOCKERFILE}" \
  --build-arg "EVALCHEMY_COMMIT=${EVALCHEMY_COMMIT}" \
  --skip-unused-stages \
  $SNAPSHOT_FLAG \
  --compressed-caching=false \
  --cache=true \
  --cache-repo="${CACHE_REPO}" \
  --destination "${DEST_PINNED}"
