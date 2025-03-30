FROM python:3.11-slim
RUN apt-get update && apt-get install -y libgl1
RUN apt-get update && apt-get install -y libglib2.0-0
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app
RUN chmod -R 777 ./datasets/data/dataset

RUN pip3 install --no-cache-dir --upgrade -r requirements.txt

USER root
RUN chmod 777 ~/app/*
USER user

EXPOSE 7860

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
