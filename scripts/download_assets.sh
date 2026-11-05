#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

mkdir -p models/wan-vae models/vjepa data third_party

download_verified() {
  local url="$1"
  local output="$2"
  local sha256="$3"
  local partial="${output}.part"
  local curl_status=0

  if [[ -f "$output" ]] && printf '%s  %s\n' "$sha256" "$output" | sha256sum -c - >/dev/null 2>&1; then
    echo "Verified, skipping: $output"
    return
  fi

  rm -f "$output"
  if [[ -f "$partial" ]] && printf '%s  %s\n' "$sha256" "$partial" | sha256sum -c - >/dev/null 2>&1; then
    mv "$partial" "$output"
    echo "Verified: $output"
    return
  fi

  curl -fL -C - --retry 20 --retry-all-errors -o "$partial" "$url" || curl_status=$?
  if [[ "$curl_status" -ne 0 ]]; then
    if [[ "$curl_status" -eq 33 ]]; then
      rm -f "$partial"
      echo "The partial file could not be resumed and was removed: $partial" >&2
    fi
    return "$curl_status"
  fi
  if printf '%s  %s\n' "$sha256" "$partial" | sha256sum -c - >/dev/null 2>&1; then
    mv "$partial" "$output"
    echo "Verified: $output"
  else
    rm -f "$partial"
    echo "Checksum verification failed: $output" >&2
    return 1
  fi
}

WAN_REV="dd9313515fdad8a9fcef162db68309090805ed30"
WAN_BASE="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/resolve/${WAN_REV}/vae"

if [[ ! -s models/wan-vae/config.json ]]; then
  curl -fL --retry 20 --retry-all-errors \
    -o models/wan-vae/config.json "$WAN_BASE/config.json"
fi
download_verified \
  "$WAN_BASE/diffusion_pytorch_model.safetensors" \
  models/wan-vae/diffusion_pytorch_model.safetensors \
  d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793

download_verified \
  https://dl.fbaipublicfiles.com/jepa/vitl16/vitl16.pth.tar \
  models/vjepa/vitl16.pth.tar \
  0cd3bfce97d04891a31eac70d573ef94f423d858a6b0dc0b3ddf5f8d00b33fe4

JEPA_COMMIT="51c59d518fc63c08464af6de585f78ac0c7ed4d5"
if [[ -d third_party/jepa/.git ]]; then
  CURRENT_COMMIT="$(git -C third_party/jepa rev-parse HEAD)"
  if [[ "$CURRENT_COMMIT" != "$JEPA_COMMIT" ]]; then
    echo "third_party/jepa is at $CURRENT_COMMIT; expected $JEPA_COMMIT." >&2
    echo "The script will not overwrite the existing checkout. Move it aside or check out the expected commit yourself." >&2
    exit 1
  fi
elif [[ -e third_party/jepa ]]; then
  echo "third_party/jepa exists but is not a Git checkout; move it aside and rerun." >&2
  exit 1
else
  git clone https://github.com/facebookresearch/jepa.git third_party/jepa
  git -C third_party/jepa checkout --detach "$JEPA_COMMIT"
fi

SPLIT_ZIP="data/UCF101TrainTestSplits-RecognitionTask.zip"
if [[ -s data/ucfTrainTestlist/trainlist01.txt && \
      -s data/ucfTrainTestlist/testlist01.txt && \
      -s data/ucfTrainTestlist/classInd.txt ]]; then
  echo "UCF101 split 1 is present, skipping split download."
else
  curl -fL -C - --retry 20 --retry-all-errors \
    -o "$SPLIT_ZIP" \
    https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip
  unzip -q -o "$SPLIT_ZIP" -d data
fi

UCF_CLASSES=(
  BasketballDunk
  GolfSwing
  LongJump
  TennisSwing
  VolleyballSpiking
  Diving
)

if python - <<'PY'
from pathlib import Path

root = Path("data")
classes = {
    "BasketballDunk", "GolfSwing", "LongJump",
    "TennisSwing", "VolleyballSpiking", "Diving",
}
names = []
for filename in ("trainlist01.txt", "testlist01.txt"):
    for line in (root / "ucfTrainTestlist" / filename).read_text().splitlines():
        name = line.split()[0]
        if name.split("/", 1)[0] in classes:
            names.append(name)
missing = [name for name in names if not (root / "UCF-101" / name).is_file()
           or (root / "UCF-101" / name).stat().st_size == 0]
print(f"UCF101 six-class files: {len(names) - len(missing)}/{len(names)} present")
raise SystemExit(1 if missing else 0)
PY
then
  echo "All six UCF101 classes are present, skipping the 6.9 GB archive download."
else
  if [[ -s data/UCF101.rar ]] && unrar t -idq data/UCF101.rar >/dev/null 2>&1; then
    echo "UCF101 archive is complete, skipping its download."
  else
    curl -fL -C - --retry 20 --retry-all-errors \
      -o data/UCF101.rar \
      https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
    unrar t -idq data/UCF101.rar >/dev/null
  fi
  for class_name in "${UCF_CLASSES[@]}"; do
    unrar x -idq -o+ data/UCF101.rar "UCF-101/${class_name}/*" data/
  done
fi

echo "Assets are ready under models/, third_party/, and data/."
