// 第 6 课 · C++ 第 1 节：第一个能编译、能运行的程序
//
// 编译：g++ hello.cpp -o hello.exe -std=c++17
// 运行：.\hello.exe
//
// 注意三点（新手最常踩的三个坑）：
//   1. endl 最后是小写字母 L（end-line 的缩写），不是数字 1
//   2. 所有引号必须是英文 " ，中文的 “ ” 编译器不认
//   3. 每条语句最后是英文分号 ;

#include <iostream>

int main() {
    std::cout << "Hello from C++" << std::endl;
    std::cout << "g++ build ok" << std::endl;
    return 0;
}
