#### 前提
zookeeper使用广泛，作为接近于开箱即用的一个服务，其日志的配置貌似没有作得很方便，接下来我们总结下这个日志配置的方法
#### 配置日志
我们的目的是配置zk的日志产生在我们设定好的目录中。
##### log4j.properties
* zookeeper日志也是使用了log4j，因此我们需要配置log4j.properties; 
* 设置
  1. 最主要是使用的appender, log输出路径，log level等
   2. 一个实例：
```
zookeeper.root.logger=INFO,ROLLINGFILE     
                                                                                                                                                                                                                                                    
zookeeper.log.dir=/log/zk/zk1 
                                                                                                                                
zookeeper.log.file=zookeeper.log                                                                                                                              
zookeeper.log.threshold=INFO                                                                                                                                  
zookeeper.log.maxfilesize=256MB                                                                                                                               
zookeeper.log.maxbackupindex=20                                                                                                                               
                                                                                                                                                              
zookeeper.tracelog.dir=${zookeeper.log.dir}                                                                                                                   
zookeeper.tracelog.file=zookeeper_trace.log                                                                                                                   
                                                                                                                                                              
log4j.rootLogger=${zookeeper.root.logger}  
#                                                                                                                                                             
# Add ROLLINGFILE to rootLogger to get log file output                                                                                                        
#                                                                                                                                                             
log4j.appender.ROLLINGFILE=org.apache.log4j.RollingFileAppender                                                                                               
log4j.appender.ROLLINGFILE.Threshold=${zookeeper.log.threshold}                                                                                               
log4j.appender.ROLLINGFILE.File=${zookeeper.log.dir}/${zookeeper.log.file}                                                                                    
log4j.appender.ROLLINGFILE.MaxFileSize=${zookeeper.log.maxfilesize}                                                                                           
log4j.appender.ROLLINGFILE.MaxBackupIndex=${zookeeper.log.maxbackupindex}                                                                                     
log4j.appender.ROLLINGFILE.layout=org.apache.log4j.PatternLayout                                                                                              
log4j.appender.ROLLINGFILE.layout.ConversionPattern=%d{ISO8601} [myid:%X{myid}] - %-5p [%t:%C{1}@%L] - %m%n 
```
##### zkEnv.sh
* zkEnv.sh是zk启动脚本的一部分，启动脚本在启动zk前执行这个zkEnv.sh来初始化设置一些启动相关的变量；
* 我们需要更改这个脚本中的`ZOO_LOG4J_PROP`和上面 log4j.properties中的配置一致，对照上面实例应该是`ZOO_LOG4J_PROP="INFO,ROLLINGFILE"`
##### zkServer.sh
* 如果你认为到这里可以万事大吉，那结果会让人失望。这里还需要一步；
* 如果执行`zkServer.sh start`来启动服务， 那你需要在zkServer.sh中找到`start`时执行的脚本代码,其中有这样一段代码：
```
nohup "$JAVA" $ZOO_DATADIR_AUTOCREATE "-Dzookeeper.log.dir=${ZOO_LOG_DIR}" \ "-Dzookeeper.log.file=${ZOO_LOG_FILE}" \                                                                                                                   
    "-Dzookeeper.root.logger=${ZOO_LOG4J_PROP}" \                                                                                                             
    -XX:+HeapDumpOnOutOfMemoryError -XX:OnOutOfMemoryError='kill -9 %p' \                                                                                     
    -cp "$CLASSPATH" $JVMFLAGS $ZOOMAIN "$ZOOCFG" > "$_ZOO_DAEMON_OUT" 2>&1 < /dev/null &
```
我们需要将其中的`"-Dzookeeper.log.dir=${ZOO_LOG_DIR}" `和`"-Dzookeeper.log.file=${ZOO_LOG_FILE}" ` 去掉。
* 到此为止，日志可以正常工作了。