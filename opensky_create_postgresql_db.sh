#!/usr/bin/env bash

set -euo pipefail

POSTGRES_FORMULA="${POSTGRES_FORMULA:-postgresql@16}"
DATABASE_NAME="${DATABASE_NAME:-opensky_scientific}"
MACOS_USERNAME="${USER:-$(id -un)}"

log() {
  printf '%s\n' "$1"
}

fail() {
  printf 'Error: %s\n' "$1" >&2
  exit 1
}

export_path_if_exists() {
  local candidate="$1"
  if [[ -d "$candidate" ]]; then
    case ":$PATH:" in
      *":$candidate:"*) ;;
      *)
        export PATH="$candidate:$PATH"
        log "Added to PATH: $candidate"
        ;;
    esac
  else
    log "Skipping PATH export because the directory does not exist: $candidate"
  fi
}

if ! command -v brew >/dev/null 2>&1; then
  fail "Homebrew is required but was not found. Install Homebrew first: https://brew.sh/"
fi

if ! brew list --versions "$POSTGRES_FORMULA" >/dev/null 2>&1; then
  log "Installing $POSTGRES_FORMULA with Homebrew..."
  brew install "$POSTGRES_FORMULA"
else
  log "$POSTGRES_FORMULA is already installed."
fi

POSTGRES_PREFIX="$(brew --prefix "$POSTGRES_FORMULA")"
POSTGRES_BIN_DIR="$POSTGRES_PREFIX/bin"
export_path_if_exists "$POSTGRES_BIN_DIR"

if ! command -v psql >/dev/null 2>&1; then
  fail "psql is still unavailable after PATH setup. Expected bin directory: $POSTGRES_BIN_DIR"
fi

log "Starting PostgreSQL service..."
brew services start "$POSTGRES_FORMULA"

if command -v pg_isready >/dev/null 2>&1; then
  log "Waiting for PostgreSQL to accept connections..."
  for _ in {1..15}; do
    if pg_isready -q; then
      break
    fi
    sleep 1
  done
fi

if ! psql postgres -tAc "SELECT 1" >/dev/null 2>&1; then
  fail "PostgreSQL is installed but not accepting local connections."
fi

DATABASE_EXISTS="$(
  psql postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname = '${DATABASE_NAME}'" \
    | tr -d '[:space:]'
)"

if [[ "$DATABASE_EXISTS" == "1" ]]; then
  log "Database already exists: $DATABASE_NAME"
else
  log "Creating database: $DATABASE_NAME"
  createdb "$DATABASE_NAME"
fi

DEFAULT_DATABASE_URL="postgresql+psycopg://${MACOS_USERNAME}@localhost/${DATABASE_NAME}"

cat <<EOF

PostgreSQL setup is complete.

Database name: $DATABASE_NAME
Local PostgreSQL user: $MACOS_USERNAME
Default SQLAlchemy URL:
  $DEFAULT_DATABASE_URL

If you want the PostgreSQL client tools available in future shells, add this to ~/.zshrc:
  export PATH="$POSTGRES_BIN_DIR:\$PATH"

Next steps:
  pipenv install psycopg[binary]
  python opensky_build_scientific_db.py build --download-dir /Volumes/YOUR_EXTERNAL_DRIVE/opensky
EOF
