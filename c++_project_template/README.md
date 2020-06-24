*  src目录：
   1. 放源码，包括.cc和.h文件，其中.h文件指的是只在本项目中使用，不用暴露出去的.h文件；
   2. src目录下可分子模块目录，比如util, network等等 ，每个子目录包含自己的.cc和.h文件；
*  include目录：
   1. 放暴露出去的.h文件；
   2. 可以分目录子模块方式；
*  3rdparty目录：
   1. 第三方依赖的库
*  build目录：
   1. 用于存放build时产生的临时文件；
*  dist
   1. 用于存放编译的最终产出，比如.a, .so或可执行文件;
   2. 分debug和release目录; 
*  examples目录：
   1. 各种事例代码
   2. build目录：编译事例代码的临时文件；
   3. dist目录： 事例代码最终可执行文件的输出目录; 
*  test目录：
   1. 单元测试代码; 
*  depoly:
   1. 打包发布脚本：package.sh; 
   2. 放置产生的最后的发布文件；
*  LICENSE
   1. license信息
*  CMakeLists.txt
   1. cmake文件；
*  build.sh
   1. 编译脚本
