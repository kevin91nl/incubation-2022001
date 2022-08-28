TESTFLAGS = ""

init:
	pip install -r requirements-dev.txt
	git init
	python -m pre_commit install
	pip install -e .
	mkdir -p tests/{unit,integration}

check:
	pydocstyle
	pydoctest
	radon cc src | grep "([6-9])\|([0-9][0-9])\|100" > /tmp/radon_output ; cat /tmp/radon_output ; [ ! -s /tmp/radon_output ] || (echo "Radon cyclomatic complexity score too high" && exit 1)
	radon mi src | grep "\- [B-F]" > /tmp/radon_output ; cat /tmp/radon_output ; [ ! -s /tmp/radon_output ] || (echo "Radon maintainability index too high" && exit 1)
	radon hal src > /tmp/radon_data && python -c "file = open('/tmp/radon_data', 'r'); data = file.read(); file.close(); lines = data.replace('\n    ', ' ').split('\n'); lines = [line for line in lines if 'bugs:' in line and float(line.split('bugs:')[-1].split()[0]) > 0.05]; file = open('/tmp/radon_output', 'w'); file.write('\n'.join(lines).strip()); file.close()" ; cat /tmp/radon_output ; [ ! -s /tmp/radon_output ] || (echo "\nRadon Halstead score too high" && exit 1)
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