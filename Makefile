.PHONY: help install test run dry-run stats alert clean

help:
	@echo "Available commands:"
	@echo "  make install    Install dependencies and Playwright browser"
	@echo "  make test       Run the complete pytest test suite"
	@echo "  make run        Execute the production job scraper pipeline"
	@echo "  make dry-run    Run pipeline without database writes or alerts"
	@echo "  make stats      Display database statistics"
	@echo "  make alert      Send a test Rich Embed alert to Discord"
	@echo "  make clean      Remove python caches and temporary files"

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	python -m playwright install chromium

test:
	pytest -v

run:
	python main.py

dry-run:
	python main.py --dry-run

stats:
	python main.py --stats

alert:
	python main.py --test-alert

clean:
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__')]"
	python -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('.pytest_cache')]"
