// 第 6 课 · C++ 第 4 节：用 ONNX Runtime 的 C 接口跑一次真推理
//
// 编译（在 src\cpp 下，把 <ORT_DIR> 换成解压出来的 onnxruntime 目录）：
//   g++ ort_detect.cpp -o ort_detect.exe -std=c++17 -O2 ^
//       -I<ORT_DIR>\include -L<ORT_DIR>\lib -lonnxruntime
//
// 运行：
//   .\ort_detect.exe ..\..\models\best.onnx input_1x3x1280x1280.bin input_meta.txt
//
// 为什么用 C 接口而不是 C++ 接口：
//   C++ 接口（onnxruntime_cxx_api.h）会把 std::string、std::vector 传进 dll，
//   而 ONNX Runtime 的 dll 是 MSVC 编的、我们用的是 MinGW —— 两边的 std 库内部结构不一样。
//   C 接口只传裸指针和整数，没有这个问题。这是 MinGW 环境下最稳的走法。
//
// 这个程序只做「推理」，不做「解码图片」：
//   输入张量由 make_input.py 先生成好（NCHW / RGB / 0~1 的 float32 裸数组）。
//   解码和缩放搬进 C++ 是下一步的事。

#include <onnxruntime_c_api.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static const OrtApi* g_ort = nullptr;

#define ORT_CHECK(expr)                                                      \
    do {                                                                     \
        OrtStatus* _s = (expr);                                              \
        if (_s != nullptr) {                                                 \
            std::fprintf(stderr, "ORT 错误 %s:%d  %s\n", __FILE__, __LINE__,  \
                         g_ort->GetErrorMessage(_s));                        \
            g_ort->ReleaseStatus(_s);                                        \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

// 把一个文件整个读进 vector<uint8_t>
static std::vector<uint8_t> read_bytes(const char* path) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        std::fprintf(stderr, "打不开文件: %s\n", path);
        std::exit(1);
    }
    fin.seekg(0, std::ios::end);
    std::streamsize n = fin.tellg();
    fin.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(n));
    fin.read(reinterpret_cast<char*>(buf.data()), n);
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::printf("用法: ort_detect.exe <model.onnx> <input.bin> <meta.txt>\n");
        return 2;
    }
    const char* model_path = argv[1];
    const char* bin_path = argv[2];
    const char* meta_path = argv[3];

    // ---------- 0. 拿到 API 表 ----------
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (g_ort == nullptr) {
        std::fprintf(stderr, "GetApi 失败\n");
        return 1;
    }
    std::printf("ONNX Runtime 版本: %s\n", OrtGetApiBase()->GetVersionString());

    // ---------- 1. 读模型（走内存，绕开 Windows 的宽字符路径） ----------
    std::vector<uint8_t> model = read_bytes(model_path);
    std::printf("模型: %s  %.1f MB\n", model_path, model.size() / 1048576.0);

    // ---------- 2. 建环境 + 会话 ----------
    OrtEnv* env = nullptr;
    ORT_CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "ort_detect", &env));

    OrtSessionOptions* opts = nullptr;
    ORT_CHECK(g_ort->CreateSessionOptions(&opts));
    ORT_CHECK(g_ort->SetIntraOpNumThreads(opts, 8));   // 和 Python 版一致
    ORT_CHECK(g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));

    OrtSession* session = nullptr;
    ORT_CHECK(g_ort->CreateSessionFromArray(env, model.data(), model.size(), opts, &session));
    std::printf("会话创建成功\n");

    // ---------- 3. 输入信息 ----------
    OrtAllocator* allocator = nullptr;
    ORT_CHECK(g_ort->GetAllocatorWithDefaultOptions(&allocator));

    size_t n_in = 0, n_out = 0;
    ORT_CHECK(g_ort->SessionGetInputCount(session, &n_in));
    ORT_CHECK(g_ort->SessionGetOutputCount(session, &n_out));

    char* input_name = nullptr;
    ORT_CHECK(g_ort->SessionGetInputName(session, 0, allocator, &input_name));
    std::printf("输入: %s   (共 %zu 个)\n", input_name, n_in);

    for (size_t i = 0; i < n_out; ++i) {
        char* on = nullptr;
        ORT_CHECK(g_ort->SessionGetOutputName(session, i, allocator, &on));
        std::printf("输出[%zu]: %s\n", i, on);
        ORT_CHECK(g_ort->AllocatorFree(allocator, on));
    }

    // ---------- 4. 读输入张量（裸 float32） ----------
    std::vector<uint8_t> raw = read_bytes(bin_path);
    const size_t n_float = raw.size() / sizeof(float);
    std::vector<float> input(n_float);
    std::memcpy(input.data(), raw.data(), n_float * sizeof(float));
    std::printf("输入张量: %zu 个 float  (%.2f MB)\n", n_float, raw.size() / 1048576.0);

    // 形状写死成 1x3x1280x1280 —— 这个值来自 make_input.py
    int64_t shape[4] = {1, 3, 1280, 1280};
    const int64_t shape_len = 4;

    OrtMemoryInfo* mem_info = nullptr;
    ORT_CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    OrtValue* input_tensor = nullptr;
    ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, input.data(), input.size() * sizeof(float),
        shape, shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));

    // ---------- 5. 跑 ----------
    const char* input_names[1] = {input_name};
    const char* output_names[1] = {"output0"};
    const OrtValue* inputs[1] = {input_tensor};

    auto t0 = std::chrono::high_resolution_clock::now();
    OrtValue* output_tensor = nullptr;
    ORT_CHECK(g_ort->Run(session, nullptr, input_names,
                         inputs, 1, output_names, 1, &output_tensor));
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("推理耗时: %.1f ms\n", ms);

    // ---------- 6. 看输出形状 ----------
    OrtTensorTypeAndShapeInfo* info = nullptr;
    ORT_CHECK(g_ort->GetTensorTypeAndShape(output_tensor, &info));
    size_t dims_count = 0;
    ORT_CHECK(g_ort->GetDimensionsCount(info, &dims_count));
    std::vector<int64_t> dims(dims_count);
    ORT_CHECK(g_ort->GetDimensions(info, dims.data(), dims_count));
    std::printf("输出形状: [");
    for (size_t i = 0; i < dims_count; ++i) {
        std::printf("%s%lld", i ? ", " : "", static_cast<long long>(dims[i]));
    }
    std::printf("]\n");
    g_ort->ReleaseTensorTypeAndShapeInfo(info);

    size_t total = 1;
    for (auto d : dims) total *= static_cast<size_t>(d);

    float* data = nullptr;
    ORT_CHECK(g_ort->GetTensorMutableData(output_tensor, reinterpret_cast<void**>(&data)));

    // ---------- 7. 解析：每行 38 个数 = x1,y1,x2,y2,score,cls + 32 个 mask 系数 ----------
    int64_t n_rows = dims[1];
    int64_t n_cols = dims[2];
    std::printf("候选框: %lld 行 × %lld 列  (总计 %zu 个数)\n",
                static_cast<long long>(n_rows),
                static_cast<long long>(n_cols), total);

    const char* kLabels[4] = {"scratch", "stain", "missing", "deform"};
    const float kConf = 0.25f;
    int kept = 0;
    for (int64_t i = 0; i < n_rows; ++i) {
        const float* row = data + i * n_cols;
        float score = row[4];
        if (score < kConf) continue;
        int cls = static_cast<int>(row[5]);
        std::printf("  #%lld  %s  conf=%.3f  box=(%.1f, %.1f, %.1f, %.1f)\n",
                    static_cast<long long>(i),
                    (cls >= 0 && cls < 4) ? kLabels[cls] : "unknown",
                    score, row[0], row[1], row[2], row[3]);
        ++kept;
    }
    std::printf("阈值 %.2f 下命中 %d 个\n", kConf, kept);

    // ---------- 8. 收拾 ----------
    // 注意：Release* 系列返回 void，不能套 ORT_CHECK 宏；AllocatorFree 返回 OrtStatus*，可以。
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(input_tensor);
    g_ort->ReleaseMemoryInfo(mem_info);
    ORT_CHECK(g_ort->AllocatorFree(allocator, input_name));
    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(opts);
    g_ort->ReleaseEnv(env);

    return kept > 0 ? 0 : 3;
}
