POETRY = poetry
VERSION_PART ?= patch
FILES ?=

ifeq ($(strip $(FILES)),)
RUFF_TARGET = .
BLACK_TARGET = .
MYPY_TARGET = src tests
PYTEST_TARGET =
else
RUFF_TARGET = $(FILES)
BLACK_TARGET = $(FILES)
MYPY_TARGET = $(FILES)
PYTEST_TARGET = $(FILES)
endif

TEST_TARGETS = $(filter tests/%,$(PYTEST_TARGET))

ifdef TEST_TARGETS
PYTEST_CMD = $(POETRY) run pytest $(TEST_TARGETS)
else ifdef PYTEST_TARGET
PYTEST_CMD = $(POETRY) run pytest $(PYTEST_TARGET)
else
PYTEST_CMD = $(POETRY) run pytest
endif

.PHONY: lint black mypy test check quick-check release publish

lint:
	$(POETRY) run ruff check .

black:
	$(POETRY) run black --check .

mypy:
	$(POETRY) run mypy src tests

test:
	$(POETRY) run pytest

check: lint black mypy test

quick-check:
	$(POETRY) run ruff check $(RUFF_TARGET)
	$(POETRY) run black $(BLACK_TARGET)
	$(POETRY) run mypy $(MYPY_TARGET)
	@exitcode=0; \
	$(PYTEST_CMD) || exitcode=$$?; \
	if [ $$exitcode -ne 0 ] && [ $$exitcode -ne 5 ]; then exit $$exitcode; fi

README.md: README.org
	emacs --batch $< -f org-md-export-to-markdown --kill

release: check
	test -z "$$(git status --porcelain)" || (echo "Working tree must be clean before releasing." >&2 && exit 1)
	@if [ -n "$(VERSION)" ]; then \
		$(POETRY) version $(VERSION); \
	elif [ -n "$(VERSION_PART)" ]; then \
		$(POETRY) version $(VERSION_PART); \
	fi
	@if ! git diff --quiet pyproject.toml src/datamat/__init__.py; then \
		VERSION=$$( $(POETRY) version --short ); \
		git add pyproject.toml src/datamat/__init__.py && \
		git commit -m "Release $$VERSION" -m "Co-authored-by: Assistant (Codex CLI)"; \
	fi
	VERSION=$$( $(POETRY) version --short ); \
	  git rev-parse -q --verify refs/tags/v$$VERSION >/dev/null || git tag v$$VERSION
	git push
	git push --tags

publish: release
	$(MAKE) README.md
	$(POETRY) build
	$(POETRY) publish --build --skip-existing --no-interaction
