FROM deepset/haystack:gpu-main

WORKDIR /run/

COPY postgresql_runner.py postgresql_runner.py

RUN pip install 'farm-haystack[milvus]'
RUN pip install 'farm-haystack[milvus2]'

ENTRYPOINT ["python", "postgresql_runner.py", "start"]
