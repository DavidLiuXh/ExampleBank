#!/bin/sh

if [ "$2" = "true" ]
then
    mvn dependency:copy-dependencies -DoutputDirectory=./lib
fi

BK_CLASSPATH=""
for i in $1/*.jar; do
    BK_CLASSPATH=${BK_CLASSPATH}:${i}
done

java -cp ${BK_CLASSPATH}:./target/test-1.0-SNAPSHOT.jar com.mytest.App
