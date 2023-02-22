# https://github.com/allenai/docker-images
# https://github.com/allenai/docker-images/pkgs/container/cuda/24038895?tag=11.2-ubuntu20.04-v0.0.15
FROM ghcr.io/allenai/cuda:11.2-ubuntu20.04-v0.0.15

COPY docker_compose_files/postgresql-docker-compose.yml docker-compose.yml

# To run the server directly:
ENTRYPOINT ["docker", "compose", "up"]
 
# To run bash:
# ENTRYPOINT ["bash", "-l"]