install-dependencies:
	pip install -r requirements.txt

install-dev-dependencies:
	pip install -r requirements-dev.txt

init:
	git init
	python -m pre_commit install
	pip install -e .

check:
	black --check .
	vulture
	pyflakes .
	pyright

test-doctest:
	python -m pytest src --doctest-modules --exitfirst

test-unit:
	python -m pytest tests/unit --exitfirst

test-integration:
	python -m pytest tests/integration --exitfirst