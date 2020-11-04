# Распределенное обучение
## Tensorflow distributed
В папке cifar_tf_distributed находятся наработки по задаче на базе tensorflow.distribute.experimental.MultiWorkerMirroredStrategy().
Оформлено в виде jupyter notebook'a. Инструкции по запуску - сформировать docker image (docker build -t tf_distributed .), запустить (docker run -it -p 8888:8888 tf_distributed), выполнить все ячейки в ноутбуке.
## Horovod
(не проверено) В папке horovod_cifar находятся наработки по задаче на базе HOROVOD. Оформлено в виде скрипта. Инструкции по запуску - сформировать docker image (docker build -t horovod_distributed .), запустить (docker run horovod_distributed).
 
