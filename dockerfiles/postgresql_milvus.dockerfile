# https://github.com/allenai/docker-images
# https://github.com/allenai/docker-images/pkgs/container/cuda/24038895?tag=11.2-ubuntu20.04-v0.0.15
FROM ghcr.io/allenai/cuda:11.2-ubuntu20.04-v0.0.15

# This needs to be passed in docker build command with --build-arg index_name=INDEX_NAME.
# NOTE: Make sure this is set after FROM, otherwise it doesn't persist.
ARG index_name

# Assert that index_name is passed
RUN test -n "$index_name"

RUN mkdir -p /net/nfs.cirrascale/aristo/harsht/postgresql_data/${index_name}
RUN mkdir -p /net/nfs.cirrascale/aristo/harsht/milvus_data/${index_name}

ENV POSTGRESQL_DATA_DIRECTORY=/net/nfs.cirrascale/aristo/harsht/postgresql_data/${index_name}
ENV MILVUS_DATA_DIRECTORY=/net/nfs.cirrascale/aristo/harsht/milvus_data/${index_name}

COPY docker_compose_files/postgresql-milvus-docker-compose.yml docker-compose.yml
COPY install_docker_compose.sh install_docker_compose.sh
RUN bash install_docker_compose.sh

# To run the server directly:
ENTRYPOINT ["docker", "compose", "up", "-d"]

# To run bash:
# ENTRYPOINT ["bash", "-l"]