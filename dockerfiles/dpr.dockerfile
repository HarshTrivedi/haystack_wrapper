FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /run/

COPY haystack_wrapper/run_name.py haystack_wrapper/run_name.py
COPY haystack_wrapper/lib.py haystack_wrapper/lib.py
COPY haystack_wrapper/train_dpr.py haystack_wrapper/train_dpr.py
COPY haystack_wrapper/index_dpr.py haystack_wrapper/index_dpr.py
COPY haystack_wrapper/predict_dpr.py haystack_wrapper/predict_dpr.py
COPY haystack_wrapper/requirements.txt requirements.txt

RUN pip install -r requirements.txt
# RUN conda install -c conda-forge jsonnet # not needed
RUN mkdir serialization_dir/

# Might need the following
# https://github.com/coqui-ai/TTS/issues/1517 (this fixed that error locally)
RUN unset LD_LIBRARY_PATH

CMD []
ENTRYPOINT []
