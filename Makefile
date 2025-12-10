.PHONY: env sanity

env:
	python -m pip install -r requirements.txt

sanity:
	@echo "Sanity check: repo structure is in place."
	@ls -R .

