// 第 6 课 · C++ 第 3 节：把 count_label 升级成真正的检查工具
//
// 编译：g++ check_label.cpp -o check_label.exe -std=c++17
// 运行：.\check_label.exe ..\..\data\yolo\labels\test\card_p022_scratch_002.txt
//
// 和 Python 版 src\data\check_labels.py 检查同样四件事：
//   1. 类别号必须在 0~3
//   2. 至少 3 个点（6 个数）
//   3. 坐标个数必须是偶数（成对出现）
//   4. 每个坐标必须落在 0~1 之间（已归一化）
//
// 返回值：全部合格 return 0；有任何一行不合格 return 1

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "usage: check_label.exe <label.txt>" << std::endl;
        return 2;
    }

    std::ifstream fin(argv[1]);
    if (!fin) {
        std::cout << "cannot open: " << argv[1] << std::endl;
        return 2;
    }

    int lineno = 0;   // 当前读到第几行
    int bad = 0;      // 有几个缺陷不合格
    std::string line;
    while (std::getline(fin, line)) {
        ++lineno;
        if (line.empty()) continue;

        std::istringstream iss(line);
        int cls = -1;
        iss >> cls;

        std::vector<double> nums;
        double v = 0.0;
        while (iss >> v) nums.push_back(v);

        // ---- 检查 1：类别号 ----
        if (cls < 0 || cls > 3) {
            std::cout << "line " << lineno << "  BAD  class=" << cls
                      << " (must be 0..3)" << std::endl;
            ++bad;
            continue;
        }

        // ---- 检查 2：至少 3 个点 ----
        if (nums.size() < 6) {
            std::cout << "line " << lineno << "  BAD  only " << nums.size() / 2
                      << " point(s), need >= 3" << std::endl;
            ++bad;
            continue;
        }

        // ---- 检查 3：坐标必须成对 ----
        if (nums.size() % 2 != 0) {
            std::cout << "line " << lineno << "  BAD  " << nums.size()
                      << " numbers (must be even)" << std::endl;
            ++bad;
            continue;
        }

        // ---- 检查 4：必须在 0~1 ----
        bool in_range = true;
        for (double x : nums) {
            if (x < 0.0 || x > 1.0) { in_range = false; break; }
        }
        if (!in_range) {
            std::cout << "line " << lineno << "  BAD  coordinate out of 0..1"
                      << std::endl;
            ++bad;
            continue;
        }

        std::cout << "line " << lineno << "  ok   class=" << cls
                  << "  points=" << nums.size() / 2 << std::endl;
    }

    std::cout << "----------------------------------------" << std::endl;
    std::cout << "lines=" << lineno << "  bad=" << bad << std::endl;
    if (bad == 0) {
        std::cout << "RESULT: PASS" << std::endl;
        return 0;
    }
    std::cout << "RESULT: FAIL" << std::endl;
    return 1;
}
