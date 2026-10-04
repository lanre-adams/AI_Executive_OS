#!/usr/bin/env bash
# Restore an AI-EOS backup created by scripts/backup.sh. THIS REPLACES CURRENT DATA.
# Usage: ./scripts/restore.sh backups/ai-eos-backup-YYYYMMDDTHHMMSSZ.tar.gz
set -euo pipefail

ARCHIVE="${1:?usage: restore.sh <backup.tar.gz>}"
[ -f "$ARCHIVE" ] || { echo "No such file: $ARCHIVE"; exit 1; }
read -r -p "This will REPLACE the database, vectors and workspace with $ARCHIVE. Type 'restore' to continue: " ok
[ "$ok" = "restore" ] || { echo "Aborted."; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
tar -xzf "$ARCHIVE" -C "$WORK"
cat "$WORK/MANIFEST" || true
PROJECT="$(docker compose config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["name"])')"

echo "==> Stopping the app"
docker compose stop app qdrant
docker compose up -d postgres redis
until docker compose exec -T postgres pg_isready -U eos -d eos >/dev/null 2>&1; do sleep 1; done

echo "==> PostgreSQL"
docker compose exec -T postgres psql -U eos -d postgres -c "DROP DATABASE IF EXISTS eos WITH (FORCE);" -c "CREATE DATABASE eos OWNER eos;"
docker compose exec -T postgres pg_restore -U eos -d eos --no-owner < "$WORK/postgres.dump"

echo "==> Qdrant"
docker run --rm -v "${PROJECT}_qdrantdata:/data" -v "$WORK:/in:ro" alpine:3.20 sh -c 'rm -rf /data/* && tar -xzf /in/qdrant.tgz -C /data'

echo "==> Workspace"
docker run --rm -v "${PROJECT}_workspace:/data" -v "$WORK:/in:ro" alpine:3.20 sh -c 'rm -rf /data/* && tar -xzf /in/workspace.tgz -C /data && chown -R 10001:10001 /data'

echo "==> Redis short-term memory is cleared (it is rebuilt automatically)"
docker compose exec -T redis redis-cli FLUSHALL >/dev/null

echo "==> Starting"
docker compose up -d
echo "Config from the backup is in $ARCHIVE (config.tgz); it was NOT applied automatically. Compare before replacing ./config."
echo "Done. Check: curl -s http://localhost:\${EOS_PORT:-8000}/health/ready"
