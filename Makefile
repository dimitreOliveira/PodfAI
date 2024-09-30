app:
	streamlit run src/app.py

build:
	pip install -r requirements.txt

lint:
	isort ./src
	black ./src
	flake8 ./src
	mypy --ignore-missing-imports ./src