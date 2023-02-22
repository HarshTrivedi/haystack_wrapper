# https://github.com/allenai/docker-images
# https://github.com/allenai/docker-images/pkgs/container/cuda/24038895?tag=11.2-ubuntu20.04-v0.0.15
FROM ghcr.io/allenai/cuda:11.2-ubuntu20.04-v0.0.15

ENV POSTGRESQL_DATA_DIRECTORY=/net/nfs.cirrascale/aristo/harsht/postgresql_data
ENV MILVUS_DATA_DIRECTORY=/net/nfs.cirrascale/aristo/harsht/milvus_data

COPY docker_compose_files/postgresql-milvus-docker-compose.yml docker-compose.yml
COPY install_docker_compose.sh install_docker_compose.sh
RUN bash install_docker_compose.sh

# To run the server directly:
ENTRYPOINT ["docker", "compose", "up"]
 
# To run bash:
# ENTRYPOINT ["bash", "-l"]