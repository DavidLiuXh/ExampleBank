#! /bin/bash

./etcd1/run.sh
sleep 1s
./etcd2/run.sh
sleep 1s
./etcd3/run.sh
