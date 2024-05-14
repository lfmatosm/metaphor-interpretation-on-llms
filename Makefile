setup:
	python -m venv ./.venv

install:
	#source ./.venv/bin/activate
	pip install -r requirements.txt

fmt:
	@autopep8 --in-place --aggressive --aggressive --max-line-length 127 src/*.py src/**/*.py scripts/*.py
	@flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics --exclude .git,__pycache__,Data
	@flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics --exclude .git,__pycache__,Data,Report,Results

gpt2:
	python -m src.main --llm gpt2-sm --temperature 0 --dataset vua_verb --task classification --seed 1 --evaluate

gpt2ft:
	python -m src.main --llm gpt2-sm --temperature 0 --dataset vua_verb --task classification --seed 1 --fine_tuning

gpt2fttest:
	python -m src.main \
		--llm gpt2-sm \
		--temperature 0 \
		--dataset vua_verb \
		--task classification \
		--fine_tuned_model_path "gpt2-sm-vua-verb-classification" \
		--seed 1 \
		--fine_tuning \
		--evaluate