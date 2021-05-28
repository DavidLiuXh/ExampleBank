* etcd简单配置，用于快速部署etcd集群
* standalone模式：
 ./etcd. --listen-client-urls=http://0.0.0.0:12379
        --listen-peer-urls=http://127.0.0.1:12380
        --advertise-client-urls=http://10.96.0.88:12379
        --data-dir=/var/etcd/data
