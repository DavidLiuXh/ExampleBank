#! /bin/bash

ps aux | grep etcd | grep -v grep | awk '{ print $2}' | xargs kill
