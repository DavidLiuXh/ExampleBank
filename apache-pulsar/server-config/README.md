* 主要是用来设置log相关
  * PULSAR_LOG_DIR=/log/pulsar/p1                                                                                                                                         
  * PULSAR_LOG_APPENDER=RollingFile                                                                                                                                       
  * PULSAR_ROUTING_APPENDER_DEFAULT=RollingFile
 
* broker.conf
   * brokerDeleteInactiveTopicsEnabled=true
      默认自动删除不活动的topic
   * brokerDeleteInactiveTopicsFrequencySeconds=60
      默认60s检查一次是不是有inactive的topic
