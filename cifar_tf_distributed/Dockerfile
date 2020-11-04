FROM python

RUN pip install tensorflow==2.3.0 jupyter numpy
COPY . /task
WORKDIR /task
ENTRYPOINT jupyter notebook --ip 0.0.0.0 --no-browser --allow-root