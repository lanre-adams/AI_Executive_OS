#!/usr/bin/env bash
# Back up everything AI-EOS needs to recover: PostgreSQL, Qdrant vectors, the agent workspace, config.
# Usage: ./scripts/backup.sh [backup-dir]        (default ./backups)
# Run from the project folder while the stack is up. Restore with ./scripts/restore.sh <archive>.
set -euo pipefail

DEST="${1:-./backups}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$DEST"
PROJECT="$(docker compose config --format json | python3 -c 'import json,sys; print(json.load(sys.stdin)["name"])')"

echo "==> PostgreSQL (logical dump)"
docker compose exec -T postgres pg_dump -U eos -d eos --format=custom --no-owner > "$WORK/postgres.dump"

echo "==> Qdrant (volume snapshot; Qdrant is paused briefly for consistency)"
docker compose stop qdrant >/dev/null
docker run --rm -v "${PROJECT}_qdrantdata:/data:ro" -v "$WORK:/out" alpine:3.20 tar -czf /out/qdrant.tgz -C /data .
docker compose start qdrant >/dev/null

echo "==> Workspace files"
docker run --rm -v "${PROJECT}_workspace:/data:ro" -v "$WORK:/out" alpine:3.20 tar -czf /out/workspace.tgz -C /data .

echo "==> Configuration (secrets in .env are NOT included - back them up separately in a password manager)"
tar -czf "$WORK/config.tgz" config

cat > "$WORK/MANIFEST" <<EOF
created_utc=$STAMP
app_version=$(docker compose exec -T app python -c 'import ai_eos; print(ai_eos.__version__)' 2>/dev/null || echo unknown)
EOF

ARCHIVE="$DEST/ai-eos-backup-$STAMP.tar.gz"
tar -czf "$ARCHIVE" -C "$WORK" .
echo "==> Done: $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"

# Keep the 14 most recent backups
ls -1t "$DEST"/ai-eos-backup-*.tar.gz 2>/dev/null | tail -n +15 | xargs -r rm -f
