POETRY = poetry
VERSION_PART ?= patch

.PHONY: lint black mypy test check release publish

lint:
	$(POETRY) run ruff check .

black:
	$(POETRY) run black --check .

mypy:
	$(POETRY) run mypy src tests

test:
	$(POETRY) run pytest

check: lint black mypy test

README.md: README.org
	emacs --batch $< -f org-md-export-to-markdown --kill

release: check
	test -z "$$(git status --porcelain)" || (echo "Working tree must be clean before releasing." >&2 && exit 1)
	$(POETRY) version $(VERSION_PART)
	VERSION=$$( $(POETRY) version --short ); \
	  git add pyproject.toml src/datamat/__init__.py && \
	  git commit -m "Release $$VERSION" -m "Co-authored-by: Assistant (Codex CLI)" && \
	  git tag v$$VERSION
	git push
	git push --tags

publish: release
	$(MAKE) README.md
	$(POETRY) build
	$(POETRY) publish --build
