FROM deepset/haystack:gpu-main

WORKDIR /run/

COPY postgresql_runner.py postgresql_runner.py
COPY docker_compose_files/ docker_compose_files/

# ENTRYPOINT ["python", "postgresql_runner.py", "start"]
ENTRYPOINT []
