**本文档针对centos系统**
* 安装开发环境
  1. yum -y groupinstall "development tools"
  2. scl enable devtoolset-8 bash
  3. 把2中的命令放到~/.bash_profile
* 在centos-dokcer里安装开发环境
  1. sudo yum install centos-release-scl
  2. sudo yum install devtoolset-[8|9d]-gcc*
  3. scl enable devtoolset-[8|9] bash
  4. 把3中的命令放到~/.bash_profile
* 编译新版linux kernel
1. download linux kernel source code;
2. 安装ncurse-devel包 （make menuconfig 文本界面窗口依赖包）
   yum -y install ncurses-devel
3. 其他依赖包
   yum -y install openssl-devel elfutils-libelf-devel bc
4. 运行 make menuconfig
5. 编译
   1. make -j n
6. 编译安装模块
   1. make modules_install
7. 编译内核核心文件
   1. make install
8. 将新版本内核设置为默认启动内核
   1. grub2-set-default 0
   2. 0表示 /boot/grub2/grub.cfg 文件中排在第一位的 menuentry 段
9. 重启
* 安装最新版tmux
  1. 一般yum直接安装的版本太低，从github直接下载最新版自行编译安装
  2. 编译依赖libevent。
* 安装pythn3
   1. sudo yum install https://repo.ius.io/ius-release-el$(rpm -E '%{rhel}').rpm
   2. sudo yum update -y
   3. sudo yum install -y python3
   4. python --version
   5. whereis python
   #设置python3为默认，可设可不设，设了yum不能用，需在其脚本中指定python版本
   6. update-alternatives --install /usr/bin/python python /usr/bin/python3.6 20
* 安装最新版vim 
1. 从官网下载源码
2. 编译：
   1. install python3-devel
   2. /configure --with-features=huge --enable-pythoninterp=yes --enable-cscope --enable-fontset --with-python-config-dir=/usr/lib64/python2.7/config --enable-python3interp=yes --with-python3-config-dir=/usr/lib64/python3.6/config-3.6m-x86_64-linux-gnu --with-python3-command=/usr/bin/python3 --enable-multibyte
   3. 配置从https://github.com/DavidLiuXh/vimrc 拉取
