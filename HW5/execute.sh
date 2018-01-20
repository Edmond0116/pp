#!/bin/bash
#INPUT_FILE=/user/ta/PageRank/Input/input-
#OUTPUT_FILE=PageRank/Output
JAR=PageRank.jar

set -xe
#hdfs dfs -rm -r $2
hadoop jar $JAR page_rank.PageRank $1 $2 $3 $4

# temp
#hdfs dfs -getmerge $2/sort hw5/test.out
#hdfs dfs -getmerge $2 homework/HW5/104021219_${DataSize}.out
