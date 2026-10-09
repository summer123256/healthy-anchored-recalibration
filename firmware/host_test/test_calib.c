/*
 * 电脑端验证片上重校准：读取 Python 导出的 calib_vectors.bin，
 * 用 C 代码逐层累加统计量、更新参数，并与 Python 的结果逐位比对；
 * 最后用校准后的模型推理测试样本，比对 logits。
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinycnn.h"
#include "model_data.h"

static int rd(void *p, size_t sz, size_t n, FILE *f) { return fread(p, sz, n, f) == n; }

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "用法: %s calib_vectors.bin\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("打开失败"); return 2; }
    char magic[4];
    uint32_t anchor, scale, nl, n, win;
    if (!rd(magic, 1, 4, f) || memcmp(magic, "TCC1", 4) || !rd(&anchor, 4, 1, f) || !rd(&scale, 4, 1, f) ||
        !rd(&nl, 4, 1, f) || !rd(&n, 4, 1, f) || !rd(&win, 4, 1, f) || win != TC_INPUT_LEN) {
        fprintf(stderr, "文件格式错误\n"); return 2;
    }
    int16_t *x = malloc(win * sizeof(int16_t));
    int8_t *q = malloc(win);
    int bad = 0;
    tc_reset_params();
    for (uint32_t k = 0; k < nl; k++) {
        tc_calib_begin();
        for (uint32_t i = 0; i < n; i++) {
            if (!rd(x, 2, win, f)) { fprintf(stderr, "读取失败\n"); return 2; }
            tc_preprocess(x, q, (uint16_t)win);
            if (tc_calib_accumulate((int)k, q) != 0) { fprintf(stderr, "累加失败\n"); return 2; }
        }
        if (tc_calib_apply((int)k, (int)anchor, (int)scale) != 0) { fprintf(stderr, "更新失败\n"); return 2; }
        uint32_t ch;
        if (!rd(&ch, 4, 1, f)) return 2;
        int32_t *eb = malloc(ch * 4), *em = malloc(ch * 4), *es = malloc(ch * 4);
        if (!rd(eb, 4, ch, f) || !rd(em, 4, ch, f) || !rd(es, 4, ch, f)) return 2;
        const int32_t *b, *m;
        const int8_t *s;
        int got = tc_get_params((int)k, &b, &m, &s);
        int layer_bad = (got != (int)ch);
        for (uint32_t c = 0; c < ch && !layer_bad; c++)
            if (b[c] != eb[c] || m[c] != em[c] || s[c] != es[c]) {
                layer_bad = 1;
                printf("  第 %u 层通道 %u 不一致: C=(%ld,%ld,%d) Python=(%ld,%ld,%ld)\n", k, c,
                       (long)b[c], (long)m[c], s[c], (long)eb[c], (long)em[c], (long)es[c]);
            }
        bad += layer_bad;
        free(eb); free(em); free(es);
    }
    uint32_t nt, nc;
    if (!rd(&nt, 4, 1, f) || !rd(&nc, 4, 1, f) || nc != TC_NUM_CLASSES) return 2;
    int32_t ref[TC_NUM_CLASSES], logits[TC_NUM_CLASSES];
    uint32_t ok = 0;
    for (uint32_t i = 0; i < nt; i++) {
        if (!rd(x, 2, win, f) || !rd(ref, 4, nc, f)) return 2;
        tc_preprocess(x, q, (uint16_t)win);
        tc_infer(q, logits);
        ok += memcmp(logits, ref, sizeof(ref)) == 0;
    }
    fclose(f);
    printf("重校准（anchor=%s, scale=%u）：%u 层参数中 %d 层不一致；校准后推理 %u/%u 条一致\n",
           anchor ? "all" : "healthy", scale, nl, bad, ok, nt);
    return (bad || ok != nt) ? 1 : 0;
}
