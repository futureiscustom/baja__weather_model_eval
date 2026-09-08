PYTHON ?= python3

.PHONY: setup test synthetic era5-data train eval grant

setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

test:
	pytest -q

synthetic:
	$(PYTHON) scripts/run_synthetic_eval.py

era5-data:
	$(PYTHON) scripts/prepare_era5_baja.py --start 2018-01-01 --end 2020-12-31

train:
	$(PYTHON) scripts/train_era5.py

eval:
	$(PYTHON) scripts/eval_era5.py

grant:
	$(PYTHON) scripts/render_grant_application.py
