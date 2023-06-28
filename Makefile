# teardown
teardown:
	make predictor_down
	make mlflow_down
	make nginx_down


#airflow for data pineline
airflow_up:
	bash run.sh  platform/airflow up	

# mlflow
mlflow_up:
	docker-compose -f platform/mlflow/docker-compose.yml up -d

mlflow_down:
	docker-compose -f platform/mlflow/docker-compose.yml down

# predictor
predictor_up:
	bash platform/deploy.sh run_predictor \
							data/model_config/phase-1/prob-1/model-1.yaml \
							data/model_config/phase-1/prob-2/model-1.yaml \
							5040

predictor_down:
	PORT=5040 docker-compose -f platform/model_predictor/docker-compose.yml down



predictor_restart:
	PORT=5040 docker-compose -f platform/model_predictor/docker-compose.yml stop
	PORT=5040 docker-compose -f platform/model_predictor/docker-compose.yml start

predictor_curl:
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-1.json
	curl -X POST http://localhost:5040/phase-1/prob-1/predict -H "Content-Type: application/json" -d @data/curl/phase-1/prob-1/payload-2.json

# using nginx for load balancing
nginx_up:
	bash platform/nginx_deploy.sh run_nginx \
							data/model_config/phase-1/prob-1/model-1.yaml \
							data/model_config/phase-1/prob-2/model-1.yaml \
							5040
nginx_down:
	PORT=5040 docker-compose -f platform/nginx/docker-compose.yml down

nginx_restart:
	PORT=5040 docker-compose -f platform/nginx/docker-compose.yml restart

