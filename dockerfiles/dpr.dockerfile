FROM deepset/haystack-gpu:latest

WORKDIR /run/

COPY run_name.py run_name.py
COPY lib.py lib.py
COPY train_dpr.py train_dpr.py
COPY index_dpr.py index_dpr.py
COPY predict_dpr.py predict_dpr.py
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt
# RUN conda install -c conda-forge jsonnet # not needed
RUN mkdir serialization_dir/

# Might need the following
# https://github.com/coqui-ai/TTS/issues/1517 (this fixed that error locally)
RUN unset LD_LIBRARY_PATH

CMD []
ENTRYPOINT []
