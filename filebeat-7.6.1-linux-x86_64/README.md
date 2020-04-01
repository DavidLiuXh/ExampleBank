* 使用单独的目录放置大量的input配置
  * 新建目录：inputs.d
  * 在filebeat.yml配置文件中添加：
  ```
  filebeat.config.inputs
    enabled: true
    path: inputs.d/*.yml
  ```
  * 在inputs.d中可添加多个input相关配置
* 动态配置reload
  * 目前只支持module和上面的inputs.d这种方式;
  * 在filebeat.yml配置文件中添加：
  ```
  filebeat.config.inputs:
  enable: true
  path: inputs.d/*.yml
  reload.enabled: true
  reload.period: 10s
  ```
* 收集的不同日志发送到不同的topic
  * 在input配置中添加`fileds`:
  ```
  fileds:
    log_topic: topic_name
  ```
  * 在output.kafka中指定：`topic: '%{[fields.log_topic]}'`
  
