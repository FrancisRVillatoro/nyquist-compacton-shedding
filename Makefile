.PHONY: check figures smoke full
check:
	python reproduce.py check
figures:
	python reproduce.py figures
smoke:
	python reproduce.py smoke
full:
	python reproduce.py full --wp all
