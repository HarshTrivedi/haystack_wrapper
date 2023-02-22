FROM deepset/haystack:gpu-main

WORKDIR /run/

COPY milvus_runner.py milvus_runner.py

RUN pip install 'farm-haystack[milvus]'
RUN pip install 'farm-haystack[milvus2]'

ENTRYPOINT ["python", "milvus_runner.py", "start"]
