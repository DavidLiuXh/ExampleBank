#! /bin/bash

ETCD_PATH=/home/lw/go/bin/etcd
ROOT_DIR=/home/lw/test/etcd/etcd3
IP=10.0.2.15

nohup ${ETCD_PATH} --initial-cluster-token etcd-cluster-test-1 \
  --name etcd3 \
  --initial-advertise-peer-urls http://${IP}:2400 \
  --listen-peer-urls http://${IP}:2400 \
  --listen-client-urls http://${IP}:2399 \
  --advertise-client-urls http://${IP}:2399 \
  --initial-cluster etcd1=http://${IP}:2380,etcd2=http://${IP}:2390,etcd3=http://${IP}:2400 \
  --initial-cluster-state new \
  --data-dir ${ROOT_DIR}/data \
  --wal-dir ${ROOT_DIR}/wal \
  --logger=zap > ${ROOT_DIR}/log/etcd.log &
 
  
