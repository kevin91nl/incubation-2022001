TESTFLAGS = ""

init:
	pip install -r requirements-dev.txt
	git init
	python -m pre_commit install
	pip install -e .

check:
	black --check .
	vulture
	pyflakes .
	pyright

-run-tests-in-folder:
	python -m pytest $(TESTDIR) $(TESTFLAGS) --exitfirst ; \
	EXIT_STATUS=$$? ; \
	[ "$$EXIT_STATUS" -eq 1 ] && exit 1 || exit 0

test-doctest: TESTDIR=src
test-doctest: TESTFLAGS=--doctest-modules
test-doctest: -run-tests-in-folder

test-unit: TESTDIR=tests/unit
test-unit: -run-tests-in-folder

test-integration: TESTDIR=tests/integration
test-integration: -run-tests-in-folder