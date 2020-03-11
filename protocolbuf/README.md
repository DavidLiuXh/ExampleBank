* 如果使用较老的protobuf v2版本，我们需要使用旧版本的protoc-gen-go来编译proto文件；
* protoc的版本无限制；
* 编译命令：protoc --go_out=[pb.go输出路径] --plugin=protoc-gen-go=[protoc-gen-go的全路径] [所需编译的proto文件]
