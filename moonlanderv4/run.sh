#!/bin/bash
# build and run docker container

docker build -t fedrl .
docker run -v $(pwd)/logs:/app/logs -v $(pwd)/plots:/app/plots fedrl
