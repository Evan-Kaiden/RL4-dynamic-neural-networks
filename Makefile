.PHONY: run

run:
	rm -rf dags/*
	python3 test_graph.py