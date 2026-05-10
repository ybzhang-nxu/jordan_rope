#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
DOWNLOAD_DIR="${ROOT_DIR}/data/downloads"
LOG_DIR="${DOWNLOAD_DIR}/logs"
QUEUE_FILE="${DOWNLOAD_DIR}/audio_datasets.aria2.txt"
SESSION_FILE="${DOWNLOAD_DIR}/audio_datasets.aria2.session"
PID_FILE="${DOWNLOAD_DIR}/audio_datasets.pid"
LOG_FILE="${LOG_DIR}/audio_datasets.log"
TMUX_SESSION="${TMUX_SESSION:-jordan_audio_downloads}"

MAX_CONCURRENT="${MAX_CONCURRENT:-2}"
CONNECTIONS="${CONNECTIONS:-4}"
INCLUDE_DEFERRED="${INCLUDE_DEFERRED:-0}"

write_queue() {
  mkdir -p "${DOWNLOAD_DIR}" "${LOG_DIR}"
  : > "${QUEUE_FILE}"

  add_item() {
    local url="$1"
    local out="$2"
    local algo="${3:-}"
    local checksum="${4:-}"

    if [[ -f "${DOWNLOAD_DIR}/${out}" && ! -f "${DOWNLOAD_DIR}/${out}.aria2" ]]; then
      if [[ -z "${algo}" ]]; then
        echo "skip existing without aria2: ${out}" >&2
        return
      fi
      if (cd "${DOWNLOAD_DIR}" && echo "${checksum}  ${out}" | "${algo}sum" -c --status -); then
        echo "skip verified: ${out}" >&2
        return
      fi
    fi

    {
      printf '%s\n' "${url}"
      printf '  out=%s\n' "${out}"
    } >> "${QUEUE_FILE}"
  }

  add_item "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip" \
    "maestro-v3.0.0-midi.zip" "sha256" \
    "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c"
  add_item "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip" \
    "fma_metadata.zip" "sha1" "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
  add_item "https://os.unil.cloud.switch.ch/fma/fma_small.zip" \
    "fma_small.zip" "sha1" "ade154f733639d52e35e32f5593efe5be76c6d70"
  add_item "https://zenodo.org/records/5120004/files/musicnet.tar.gz?download=1" \
    "musicnet.tar.gz" "md5" "844764911fa0d5b97c97da944a057590"

  if [[ "${INCLUDE_DEFERRED}" == "1" ]]; then
    add_item "https://zenodo.org/records/5120004/files/musicnet_metadata.csv?download=1" \
      "musicnet_metadata.csv" "md5" "1caef62cee9c875235e62aac368b49d8"
    add_item "https://zenodo.org/records/5120004/files/musicnet_midis.tar.gz?download=1" \
      "musicnet_midis.tar.gz" "md5" "b5fa98a113bfc51c8a445def9f24dc7e"
    add_item "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz" \
      "nsynth-train.jsonwav.tar.gz"
    add_item "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz" \
      "nsynth-valid.jsonwav.tar.gz"
    add_item "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz" \
      "nsynth-test.jsonwav.tar.gz"
    add_item "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1" \
      "slakh2100_flac_redux.tar.gz" "md5" "f4b71b6c45ac9b506f59788456b3f0c4"
    add_item "https://zenodo.org/records/1117372/files/musdb18.zip?download=1" \
      "musdb18.zip" "md5" "af06762477334799bfc5abf237648207"
  fi

  if [[ ! -s "${QUEUE_FILE}" ]]; then
    echo "all queued files are already present and verified" >&2
  fi
}

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

start_download() {
  write_queue
  if [[ ! -s "${QUEUE_FILE}" ]]; then
    exit 0
  fi
  if is_running; then
    echo "already running: pid $(cat "${PID_FILE}")"
    exit 0
  fi

  if command -v tmux >/dev/null 2>&1; then
    if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
      echo "tmux session already exists: ${TMUX_SESSION}"
      exit 0
    fi
    tmux new-session -d -s "${TMUX_SESSION}" "cd '${ROOT_DIR}' && '${SCRIPT_PATH}' run"
    echo "started in tmux: ${TMUX_SESSION}"
    echo "log: ${LOG_FILE}"
    echo "queue: ${QUEUE_FILE}"
    return
  fi

  echo "tmux not found; running in the foreground" >&2
  run_download
}

run_download() {
  mkdir -p "${DOWNLOAD_DIR}" "${LOG_DIR}"
  echo "$$" > "${PID_FILE}"
  exec aria2c \
    --dir="${DOWNLOAD_DIR}" \
    --input-file="${QUEUE_FILE}" \
    --save-session="${SESSION_FILE}" \
    --save-session-interval=60 \
    --continue=true \
    --max-concurrent-downloads="${MAX_CONCURRENT}" \
    --max-connection-per-server="${CONNECTIONS}" \
    --split="${CONNECTIONS}" \
    --min-split-size=1M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --retry-wait=30 \
    --max-tries=0 \
    --summary-interval=60 \
    --console-log-level=notice \
    > "${LOG_FILE}" 2>&1
}

status_download() {
  if is_running; then
    echo "running: pid $(cat "${PID_FILE}")"
  else
    echo "not running"
  fi
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    echo "tmux session: ${TMUX_SESSION}"
  fi
  echo
  echo "download files:"
  find "${DOWNLOAD_DIR}" -maxdepth 1 -type f \
    \( -name '*.zip' -o -name '*.tar.gz' -o -name '*.csv' -o -name '*.aria2' \) \
    -printf '%f\t%s bytes\t%TY-%Tm-%Td %TH:%TM\n' | sort
  echo
  if [[ -f "${LOG_FILE}" ]]; then
    echo "log tail:"
    tail -n 40 "${LOG_FILE}"
  fi
}

stop_download() {
  if is_running; then
    kill -INT "$(cat "${PID_FILE}")"
    echo "sent SIGINT to pid $(cat "${PID_FILE}")"
  else
    echo "not running"
  fi
  if command -v tmux >/dev/null 2>&1 && tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
    tmux kill-session -t "${TMUX_SESSION}"
    echo "closed tmux session: ${TMUX_SESSION}"
  fi
}

verify_downloads() {
  cd "${DOWNLOAD_DIR}"
  echo "70470ee253295c8d2c71e6d9d4a815189e35c89624b76d22fce5a019d5dde12c  maestro-v3.0.0-midi.zip" | sha256sum -c -
  echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
  echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip" | sha1sum -c -
  echo "844764911fa0d5b97c97da944a057590  musicnet.tar.gz" | md5sum -c -
  if [[ "${INCLUDE_DEFERRED}" == "1" ]]; then
    cat <<'MD5SUMS' | md5sum -c -
1caef62cee9c875235e62aac368b49d8  musicnet_metadata.csv
b5fa98a113bfc51c8a445def9f24dc7e  musicnet_midis.tar.gz
af06762477334799bfc5abf237648207  musdb18.zip
f4b71b6c45ac9b506f59788456b3f0c4  slakh2100_flac_redux.tar.gz
MD5SUMS
  fi
}

case "${1:-start}" in
  start)
    start_download
    ;;
  run)
    run_download
    ;;
  status)
    status_download
    ;;
  stop)
    stop_download
    ;;
  queue)
    write_queue
    echo "${QUEUE_FILE}"
    ;;
  verify)
    verify_downloads
    ;;
  *)
    echo "usage: $0 {start|status|stop|queue|verify}" >&2
    exit 2
    ;;
esac
