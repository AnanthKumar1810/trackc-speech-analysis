.PHONY: all test validate tune evaluate nuisance viewer report serve

all: validate test tune evaluate nuisance viewer report

test:
	python -m unittest discover -s src -p 'test_*.py' -v

validate:
	python src/validate.py

tune:
	python src/tune_localizer.py

evaluate:
	python src/evaluate_localizer.py

nuisance:
	python src/stress_test_nuisance.py

viewer:
	python src/make_viewer_data.py

report:
	python src/render_technical_pdf.py

serve:
	python src/serve.py
