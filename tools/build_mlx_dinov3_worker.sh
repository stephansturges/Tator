#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/tools/mlx_dinov3_worker"

# This dependency tree is public. Some local developer machines rewrite GitHub
# HTTPS URLs to SSH globally, which breaks unattended SwiftPM resolution.
export GIT_CONFIG_GLOBAL="${GIT_CONFIG_GLOBAL:-/dev/null}"

swift build -c release --product mlx-dinov3-worker
swift build -c release --product mlx-dinov3-convert

mount_metal_cryptex() {
  if [[ -x "/Volumes/MetalToolchainCryptex/Metal.xctoolchain/usr/bin/metal" ]]; then
    return 0
  fi
  local dmg
  dmg="$(find /System/Library/AssetsV2/com_apple_MobileAsset_MetalToolchain -path '*/AssetData/Restore/*.dmg' -print 2>/dev/null | sort | tail -1 || true)"
  if [[ -n "$dmg" ]]; then
    hdiutil attach -nobrowse -readonly "$dmg" >/dev/null 2>&1 || true
  fi
}

find_tool() {
  local name="$1"
  local candidate
  candidate="$(xcrun --find "$name" 2>/dev/null || true)"
  if [[ -n "$candidate" ]] && "$candidate" -v >/dev/null 2>&1; then
    echo "$candidate"
    return 0
  fi
  mount_metal_cryptex
  candidate="/Volumes/MetalToolchainCryptex/Metal.xctoolchain/usr/bin/$name"
  if [[ -x "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi
  echo "Missing usable $name. Run xcodebuild -downloadComponent MetalToolchain, then mount the downloaded MetalToolchainCryptex dmg if Xcode cannot remount it." >&2
  return 1
}

METAL_BIN="${MLX_DINOV3_METAL_BIN:-$(find_tool metal)}"
METALLIB_BIN="${MLX_DINOV3_METALLIB_BIN:-$(find_tool metallib)}"
CMLX_ROOT="$ROOT/tools/mlx_dinov3_worker/.build/checkouts/mlx-swift/Source/Cmlx"
AIR_DIR="$ROOT/tools/mlx_dinov3_worker/.build/mlx_metallib_air"
RELEASE_DIR="$ROOT/tools/mlx_dinov3_worker/.build/release"

rm -rf "$AIR_DIR"
mkdir -p "$AIR_DIR"
(
  cd "$CMLX_ROOT"
  while IFS= read -r source; do
    output="$AIR_DIR/${source//\//_}.air"
    "$METAL_BIN" -c "$source" \
      -I mlx/mlx/backend/metal/kernels \
      -I mlx/mlx/backend/metal \
      -I mlx \
      -o "$output"
  done < <(find mlx-generated/metal -name '*.metal' | sort)
)
"$METALLIB_BIN" "$AIR_DIR"/*.air -o "$RELEASE_DIR/mlx.metallib"
cp "$RELEASE_DIR/mlx.metallib" "$RELEASE_DIR/default.metallib"

echo "$RELEASE_DIR/mlx-dinov3-worker"
