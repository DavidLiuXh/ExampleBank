[toc]

#### 使用mvn创建java工程

#### 安装mvn

我们之前参考官网就好: [Installing Apache Maven](http://maven.apache.org/install.html)

##### 创建java工程

* 命令行执行

  ```
  mvn archetype:generate -DgroupId=com.mytest -DartifactId=test -DarchetypeGroupId=org.apache.maven.archetypes -DarchetypeArtifactId=maven-archetype-quickstart -DarchetypeVersion=1.4  -DinteractiveMode=false
  ```

  至少需要提供 `groupId 包名`和` artifactId 工程名也就是jar包的名称`。

* 生成目录结构

  ```
  ./
  ├── pom.xml
  └── src
      ├── main
      │   └── java
      │       └── com
      │           └── mytest
      │               └── App.java
      └── test
          └── java
              └── com
                  └── mytest
                      └── AppTest.java
  ```

##### pom.xml配置文件

* 如果需要使用到java 8的特性，比如lambda表达式，需要调整 `maven.compiler.source`和 `mavin.compiler.target`到`1.8`或更高。

  ```
  <maven.compiler.source>1.8</maven.compiler.source>                           <maven.compiler.target>1.8</maven.compiler.target>
  ```

* 如果你的工程需要第三方依赖，需添加依赖到`<dependencies></dependencies>`之间

  ```
  <dependencies>
      <dependency>          
        <groupId>junit</groupId>       
        <artifactId>junit</artifactId>       
        <version>4.11</version>       
        <scope>test</scope>          
      </dependency>            
    </dependencies> 
  ```

* 如果需要依赖本地的jar包，而非mvn库中的，则需要使用如下形式，需要`scope`和`systempath`属性加持。

  ```
     <!--使用本地jar包，非mvn库中的-->
     <dependency>
       <groupId>org.apache.bookkeeper</groupId>
        <artifactId>bookkeeper-server</artifactId>
        <version>${bookkeeper.version}</version>
        <scope>system</scope>
        <systemPath>${bookkeeper.localjar}</systemPath>
     </dependency>   
  ```

* 对于最终编译产生的jar, 在通过`java -jar `执行时，可通过pom.xml指定生成的jar中的主类

  如果不配置这个，在最后执行的时候需要加上类名：`java -jar xxxx com.mytest.App`

  ```
  <!--指定生成的jar中的主类-->
            <plugin>
              <groupId>org.apache.maven.plugins</groupId>
              <artifactId>maven-jar-plugin</artifactId>
              <version>3.0.2</version>
              <configuration>
              <archive>
                  <manifest>
                      <addClasspath>true</addClasspath>
                          <mainClass>com.mytest.App</mainClass>
                          </manifest>
                      </archive>
                  </configuration>
              </plugin>         
  ```

##### 编译

* 在`pom.xml`所在目录下执行 `mvn clean package`
* 编译完成生成 `target`目录， 生成的jar包类似： test-1.0-SNAPSHOT.jar

##### 执行

* 如果有第三方依赖，在编译时可以选择将第三方依赖全部打包到最终的jar包中；

* 我们这里选择另外一种方式，执行时执定class path的方式： `java -cp xxxx:xxxx`

* 我们需要将所有的第三方依赖的jar包集中到一起:

  1. 创建一个名为`lib`的目录；

  2. `mvn dependency:copy-dependencies -DoutputDirectory=[上面创建的lib目录的全路径]`

     这条命令将把在`pom.xml`文件中`dependencies`下面的依赖的jar包自动拷贝到lib目录下；

* 执行

  下面给出了一个简单的脚本, 先收集lib下面所有的jar包作为`-cp`的参数

  ```
  BK_CLASSPATH=""                                                                                                                                         #收集lib下面所有的jar包作为`-cp`的参数 
  for i in $1/*.jar; do
      BK_CLASSPATH=${BK_CLASSPATH}:${i}
  done   
  
  #运行test-1.0-SNAPSHOT.jar
  java -cp ${BK_CLASSPATH}:/home/xx/test/java/test/target/test-1.0-SNAPSHOT.jar com.mytest.App
  ```

  