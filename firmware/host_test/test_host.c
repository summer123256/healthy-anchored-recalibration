/*
 * 电脑端验证程序：用 gcc 编译单片机的推理代码，逐条比对 Python 导出的测试向量。
 * 只要这里全部通过，说明 C 代码与 Python 整数参考实现逐位一致。
 *
 * 编译运行（在 firmware/host_test 目录）：
 *   make MODEL_DIR=../../python/export/tiny   && ./test_host ../../python/export/tiny/vectors.bin
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinycnn.h"
#include "model_data.h"

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "用法: %s vectors.bin\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("打开失败"); return 2; }
    char magic[4];
    uint32_t n, len, ncls;
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TCV1", 4) != 0 ||
        fread(&n, 4, 1, f) != 1 || fread(&len, 4, 1, f) != 1 || fread(&ncls, 4, 1, f) != 1) {
        fprintf(stderr, "文件格式错误\n"); return 2;
    }
    if (len != TC_INPUT_LEN || ncls != TC_NUM_CLASSES) {
        fprintf(stderr, "维度不一致: 文件 %u/%u, 模型 %d/%d\n", len, ncls, TC_INPUT_LEN, TC_NUM_CLASSES);
        return 2;
    }
    int16_t *x = malloc(len * sizeof(int16_t));
    int32_t *ref = malloc(ncls * sizeof(int32_t));
    int8_t *q = malloc(len);
    int32_t logits[TC_NUM_CLASSES];
    uint32_t ok = 0, bad = 0;
    tc_reset_params();                 /* 上电后必须调用：把卷积层参数从 Flash 复制到 RAM */
    for (uint32_t i = 0; i < n; i++) {
        if (fread(x, sizeof(int16_t), len, f) != len || fread(ref, sizeof(int32_t), ncls, f) != ncls) {
            fprintf(stderr, "读取第 %u 条失败\n", i); return 2;
        }
        tc_preprocess(x, q, (uint16_t)len);
        tc_infer(q, logits);
        int same = 1;
        for (uint32_t c = 0; c < ncls; c++) if (logits[c] != ref[c]) same = 0;
        if (same) ok++;
        else if (bad++ < 5) {
            printf("不一致 #%u: C=", i);
            for (uint32_t c = 0; c < ncls; c++) printf("%ld ", (long)logits[c]);
            printf(" Python=");
            for (uint32_t c = 0; c < ncls; c++) printf("%ld ", (long)ref[c]);
            printf("\n");
        }
    }
    fclose(f);
    printf("模型 %s: %u 条测试向量，一致 %u 条，不一致 %u 条\n", TC_MODEL_NAME, n, ok, bad);
    printf("激活缓冲区: 2 x %d 字节\n", TC_MAX_ACT);
    return bad ? 1 : 0;
}
