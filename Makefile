project=$(notdir $(shell pwd))
host=$(shell hostname -I | grep -o ^[0-9.]*)

.PHONY: build_dev build_prod setup start_dev start_prod start_dev_tf clean

build_dev:
	docker build -t $(project)_dev:latest -f Dockerfile_dev .

build_prod:
	docker build -t $(project)_prod:latest -f Dockerfile_prod .

start_dev:
	$(eval token=$(shell bash -c 'read -p "Choose Jupyter token (visible for everyone with access to the server): " token; echo $$token'))
	$(eval port=$(shell (python3 /projects/.setup/docker.py $(USER) $(project)) 2>&1))

	@-docker stop $(USER)-$(project)_dev > /dev/null 2>&1 ||:
	@-docker container prune --force > /dev/null

	@docker run \
		-u $(shell id -u):$(shell id -g) -it -d \
		-p $(port):8888 $(tf)\
		--rm \
		--name $(USER)-$(project)_dev \
		-e PYTHONPATH=/app/src/utils \
		--cpus=6 \
		-v $(PWD)/:/app/ $(project)_dev:latest bash > /dev/null

	@docker exec -it -d $(USER)-$(project)_dev bash \
		-c "jupyter lab --ip 0.0.0.0 --no-browser --NotebookApp.token=$(token)"

	@echo "Container started"
	@echo "Jupyter is running at http://$(host):$(port)/?token=$(token)"

start_prod:
	@-docker stop $(project)_prod > /dev/null 2>&1 ||:
	@-docker container prune --force > /dev/null

	docker run -t\
		--name $(project)_prod \
		-e PYTHONPATH=/app/src/utils \
		-e CREDENTIALS_SQL_USR=$(CREDENTIALS_SQL_USR) \
		-e CREDENTIALS_SQL_PSW=$(CREDENTIALS_SQL_PSW) \
		$(project)_prod:latest bash -c "python /logger/docker_stat_recorder.py /app/src/predict_model.py"
	docker cp $(project)_prod:/logger stats_log_html
	docker rm $(project)_prod

start_dev_tf:
	$(eval tf_port=$(shell (python3 /projects/.setup/docker.py $(USER) $(project)_tensorboard) 2>&1))
	@$(MAKE) -s start_dev tf="-p $(tf_port):6006 --gpus all"
	@echo "Tensorboard is available at http://$(host):$(tf_port). Enter container with docker exec -it $(USER)-$(project)_dev bash and start Tensorboard manually"

clean:
	docker container prune --force
	docker image prune --force