#!/bin/sh

CRTDIR=$(pwd)
BK_CLASSPATH=""
for i in $CRTDIR/lib/*.jar; do
    BK_CLASSPATH=${BK_CLASSPATH}:${i}
done

java -cp ${BK_CLASSPATH}:./target/pulsar-test-1.0-SNAPSHOT.jar -Dlog4j.configuration=file:$CRTDIR/log4j.properties  -Dlog4j.debug com.mytest.App $1
