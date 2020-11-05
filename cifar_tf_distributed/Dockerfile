FROM python

ENV CUDA_VISIBLE_DEVICES=-1
EXPOSE 12345 23456 34567
RUN pip install tensorflow==2.3.0 jupyter numpy
COPY . /task
WORKDIR /task
ENTRYPOINT sh commands.sh