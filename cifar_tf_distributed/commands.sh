#!/bin/bash
nohup python main.py 0 &> job_0.log
sleep 10
cat job_0.log
nohup python main.py 1 &> job_1.log
sleep 10
cat job_1.log
python main.py 2
cat job_0.log
cat job_1.log