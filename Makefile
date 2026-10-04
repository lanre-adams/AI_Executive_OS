.PHONY: help install dev test cov lint format up down logs ps backup restore admin secrets clean

help:            ## Show this help
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

install:         ## Install for local development (virtualenv recommended)
	pip install -e ".[dev,analytics]"

dev:             ## Run the API + dashboard locally with auto-reload
	eos serve --reload

test:            ## Run the test suite
	pytest

cov:             ## Run tests with coverage (fails under 90%)
	pytest --cov=ai_eos --cov-report=term-missing --cov-fail-under=90

lint:            ## Lint and type-check
	ruff check src tests && ruff format --check src tests && mypy src

format:          ## Auto-format
	ruff format src tests && ruff check --fix src tests

up:              ## Start the Docker stack
	docker compose up -d --build

down:            ## Stop the Docker stack (data is kept)
	docker compose down

logs:            ## Follow app logs
	docker compose logs -f app

ps:              ## Show container status
	docker compose ps

backup:          ## Back up database, vectors and workspace to ./backups
	./scripts/backup.sh

restore:         ## Restore: make restore FILE=backups/ai-eos-backup-....tar.gz
	./scripts/restore.sh $(FILE)

admin:           ## Create an admin user inside the running container
	docker compose exec app eos create-user --role admin

secrets:         ## Generate EOS_JWT_SECRET and EOS_ENCRYPTION_KEY values
	@python -c "import secrets; print('EOS_JWT_SECRET=' + secrets.token_urlsafe(48))"
	@python -c "from cryptography.fernet import Fernet; print('EOS_ENCRYPTION_KEY=' + Fernet.generate_key().decode())"

clean:           ## Remove caches
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
