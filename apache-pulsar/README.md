* test backlog policy                                                                                                                                                                                     
   * ./pulsar-admin namespaces set-backlog-quota --limit 100000 --policy producer_exception my-tenants-1/my-namespace-1                                                                                   
   * http://pulsar.apache.org/docs/en/admin-api-namespaces/                                                                                                                                                  
   *  在Create Producer和 send 时都可能抛出ProducerBlockedQuotaExceededException                                                                                                                               
   *  send中抛出时，如果调整 limit，send可以恢复，不用重启producer
 * Partitioned topic
    * 如果生产时的Msg没有提供Key, 切使用 RoundRobinPartition 策略，则消费时不能保证顺序
    * 如果希望保证消费的Msg顺序
       * Msg附带key, 相同key的Msg被hash到相同的partition(即子topic)上；
       * 使用非Partitioned topic, 消费时使用非shard方式；若使用shard方式，可考虑用shard_Key方式; 
