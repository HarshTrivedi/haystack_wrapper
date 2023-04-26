FROM python:latest

WORKDIR /run/
COPY occupy_cpu.py occupy_cpu.py

ENTRYPOINT []
CMD ["python", "/run/occupy_cpu.py"]
