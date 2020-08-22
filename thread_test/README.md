#### linux signal简述
* linux中的signal机制，应用层可以通过 `kill`，触发系统调用来发送signal到相应的task; 
* signal不会立即被处理，它会进入到task的pending signal队列；
* task在系统调用从kernal返回用户空间时，会检查这个pending signal，处理完所有的signal后，再返回到用户容间；
* 应用层可以设置自己的signal处理函数，但这个函数只能在用户空间执行，而signal被处理是在返回用户空间前，因此kernel需在用户栈上创建一个新的栈帧来执行用户自定义的处理函数。
#### 利用signal机制实现pthread_cancel
* pthread_cancel发送终止信息给thread, thread的运行实体函数在系统调用返回时处理这个signal，退出;
* 如果thread运行实体中没有系统调用，则pthread_cancel将无法终止该thread;
* 如何解决上面的问题呢？在thread实体中加入`pthread_testcancel()`。
