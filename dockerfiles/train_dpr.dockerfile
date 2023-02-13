FROM deepset/haystack:gpu-main

WORKDIR /run/

COPY train_dpr.py train_dpr.py

RUN pip install dictparse
RUN pip install 'farm-haystack[milvus]'
RUN pip install 'farm-haystack[milvus2]'

RUN conda install -c conda-forge jsonnet
RUN mkdir serialization_dir/

# Might need the following
# https://github.com/coqui-ai/TTS/issues/1517 (this fixed that error locally)
RUN unset LD_LIBRARY_PATH

CMD []
ENTRYPOINT []
