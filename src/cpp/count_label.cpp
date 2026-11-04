// 第 6 课 · C++ 第 2 节：读一个 YOLO 标注文件，数出有几个缺陷、每个缺陷几个点
//
// 编译：g++ count_label.cpp -o count_label.exe -std=c++17
// 运行：.\count_label.exe ..\..\data\yolo\labels\test\card_p022_scratch_002.txt
//
// 这个程序做的事，和 Python 的 src\data\check_labels.py 是同一件：
// 读一个 txt，检查它格式对不对。同一个任务，两种语言。

#include <fstream>    // ifstream：读文件
#include <iostream>   // cout / endl
#include <sstream>    // istringstream：把一行拆成一个个数
#include <string>     // std::string
#include <vector>     // std::vector：能自动变长的数组

int main(int argc, char** argv) {
    // ---------- 1. 检查有没有给文件路径 ----------
    if (argc < 2) {
        std::cout << "usage: count_label.exe <label.txt>" << std::endl;
        return 1;   // 返回非 0 表示出错
    }
    std::string path = argv[1];

    // ---------- 2. 打开文件 ----------
    std::ifstream fin(path);
    if (!fin) {
        std::cout << "cannot open: " << path << std::endl;
        return 1;
    }

    // ---------- 3. 一行一行读 ----------
    std::vector<std::vector<double>> shapes;   // 每个缺陷 = 一堆坐标
    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;            // 空行跳过

        std::istringstream iss(line);
        int cls = -1;
        iss >> cls;                            // 第一个数是类别号

        std::vector<double> nums;
        double v = 0.0;
        while (iss >> v) nums.push_back(v);    // 剩下的全是坐标

        shapes.push_back(nums);
    }

    // ---------- 4. 统计并打印 ----------
    std::cout << "file  : " << path << std::endl;
    std::cout << "lines : " << shapes.size() << "  (defects)" << std::endl;
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        std::cout << "  #" << (i + 1)
                  << "  points = " << shapes[i].size() / 2 << std::endl;
    }
    return 0;
}
