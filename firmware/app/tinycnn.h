/*
 * tinycnn.h —— 面向 Cortex-M3（无 FPU）的 int8 一维卷积网络推理与片上重校准
 *
 * 纯 C99、只用整数运算；与 Python 端 quant.int_forward()、preprocess.preprocess_int16()、
 * recal.recalibrate() 逐位一致。
 *
 * 片上重校准（健康状态锚定）：
 *   卷积层的偏置、重量化乘数、移位在启动时从 Flash 复制到 RAM（tc_reset_params），
 *   校准时逐层统计现场数据的 Σacc、Σacc²，再与 Flash 中的源域统计量对齐，更新 RAM 中的参数。
 */
#ifndef TINYCNN_H
#define TINYCNN_H

#include <stdint.h>

enum { TC_CONV = 0, TC_FC = 1, TC_MAXPOOL = 2, TC_GAP = 3, TC_FLATTEN = 4 };
enum { TC_ANCHOR_HEALTHY = 0, TC_ANCHOR_ALL = 1 };

typedef struct {
    uint8_t  type;        /* TC_CONV / TC_FC / TC_MAXPOOL / TC_GAP / TC_FLATTEN */
    uint8_t  relu;        /* 1: 输出下限 0 */
    uint8_t  last;        /* 1: 最后一层，输出 int32 logits */
    uint16_t in_ch, in_len, out_ch, out_len;
    uint16_t kernel, stride, pad, groups;
    uint16_t ch_off;      /* 可校准卷积层：在 RAM 参数表中的起始通道号 */
    const int8_t  *w;     /* conv: [out_ch][in_ch/groups][kernel]; fc: [out][in] */
    const int32_t *b;     /* Flash 中的原始偏置 [out_ch] */
    const int32_t *mult;  /* Flash 中的原始重量化乘数 [out_ch] */
    const int8_t  *shift; /* Flash 中的原始移位 [out_ch] */
    const int32_t *mu_h, *sig_h;  /* 源域健康类统计量（可为 0） */
    const int32_t *mu_a, *sig_a;  /* 源域全部类别统计量（可为 0） */
} tc_layer_t;

/* 四舍五入整数除法（远离零），b > 0 */
static inline int32_t tc_div_round(int32_t a, int32_t b)
{
    return (a >= 0) ? (a + b / 2) / b : -((-a + b / 2) / b);
}

/* ---------------- 推理 ---------------- */
void tc_preprocess(const int16_t *x, int8_t *out, uint16_t len);
int  tc_infer(const int8_t *in, int32_t *logits);   /* 返回预测类别 */

/* ---------------- 片上重校准 ---------------- */
void tc_reset_params(void);          /* 恢复出厂参数（Flash → RAM），上电时必须调用一次 */
int  tc_num_calib_layers(void);      /* 可校准卷积层个数 */
void tc_calib_begin(void);           /* 清零当前层的统计累加器 */
/* 用一个窗口累加第 k 个可校准卷积层的统计量；返回 0 成功 */
int  tc_calib_accumulate(int k, const int8_t *in);
/* 用累加结果更新第 k 层参数；anchor: TC_ANCHOR_HEALTHY / TC_ANCHOR_ALL；scale: 1 校正尺度 */
int  tc_calib_apply(int k, int anchor, int scale);
/* 读取第 k 层的当前参数（用于与 Python 逐位比对）；返回通道数 */
int  tc_get_params(int k, const int32_t **b, const int32_t **mult, const int8_t **shift);

#endif
