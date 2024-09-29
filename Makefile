app:
	streamlit run src/app.py

lint:
	isort ./src
	black ./src
	flake8 ./src
	mypy --ignore-missing-imports ./src