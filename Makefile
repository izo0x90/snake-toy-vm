SHELL := /bin/bash
# Target machine

.PHONY: lint
lint:
	ruff check .

.PHONY: lint-fix
lint-fix:
	ruff check --fix .

.PHONY: format
format:
	ruff format

.PHONY: type-check
type-check:
	pyright .

.PHONY: check-all
check-all: format lint-fix type-check

.PHONY: freeze-deps
freeze-deps:
	uv pip freeze > requirements.txt

.PHONY: todo
todo:
	@less -f <(grep -r -I --exclude=Makefile --exclude-dir=.ruff_cache \
	--exclude-dir=.venv --exclude-dir=examples 'TODO:' .) \
	<(cat ./TODO.MD)

.PHONY: dev_build_env
dev_build_env:
	([ ! -f .venv/bin/activate ] && uv venv \
	&& . .venv/bin/activate && uv pip install -r requirements.txt \
	) || echo "Venv already exists"
	([ ! -f dev_env ] && ln -s .venv/bin/activate dev_env) || \
	echo "Venv link already exists"
