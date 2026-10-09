/*
 * tinycnn.c —— int8 推理引擎与片上健康状态锚定重校准
 *
 * RAM：两个 TC_MAX_ACT 字节的激活缓冲区（乒乓）；
 *      可校准卷积层的偏置/乘数/移位副本（TC_CALIB_CH 个通道，每通道 9 字节）；
 *      统计累加器（TC_MAX_CALIB_CH 个通道，每通道 16 字节）。
 * 重量化：out = (acc * M + 2^(s-1)) >> s，使用 Cortex-M3 的 SMULL 64 位乘法。
 */
#include "tinycnn.h"
#include "model_data.h"

static int8_t s_buf_a[TC_MAX_ACT];
static int8_t s_buf_b[TC_MAX_ACT];

/* 可校准卷积层的运行时参数（RAM） */
static int32_t s_bias[TC_CALIB_CH];
static int32_t s_mult[TC_CALIB_CH];
static int8_t  s_shift[TC_CALIB_CH];

/* 统计累加器 */
static int64_t  s_sum1[TC_MAX_CALIB_CH];
static int64_t  s_sum2[TC_MAX_CALIB_CH];
static uint32_t s_cnt;

#define CAL_Q      16
#define CAL_ONE    ((int64_t)1 << CAL_Q)
#define CAL_AQ_MIN ((int64_t)1 << (CAL_Q - 2))
#define CAL_AQ_MAX ((int64_t)1 << (CAL_Q + 2))

static inline int is_calib_layer(const tc_layer_t *L) { return L->type == TC_CONV && !L->last; }

static int64_t div_round64(int64_t a, int64_t b)
{
    return (a >= 0) ? (a + b / 2) / b : -((-a + b / 2) / b);
}

static uint64_t isqrt64(uint64_t n)            /* 向下取整的整数平方根 */
{
    uint64_t res = 0, bit = (uint64_t)1 << 62;
    while (bit > n) bit >>= 2;
    while (bit) {
        if (n >= res + bit) { n -= res + bit; res = (res >> 1) + bit; }
        else res >>= 1;
        bit >>= 2;
    }
    return res;
}

static inline int8_t requant(int32_t acc, int32_t mult, int8_t shift, uint8_t relu)
{
    int64_t v = (int64_t)acc * mult + ((int64_t)1 << (shift - 1));
    v >>= shift;                     /* 算术右移（GCC 对有符号数为算术移位） */
    const int64_t lo = relu ? 0 : -127;
    if (v < lo) v = lo;
    if (v > 127) v = 127;
    return (int8_t)v;
}

/* ================================================================ 预处理 */
void tc_preprocess(const int16_t *x, int8_t *out, uint16_t len)
{
    int32_t sum = 0;
    for (uint16_t i = 0; i < len; i++) sum += x[i];
    const int32_t mean = tc_div_round(sum, (int32_t)len);

    int32_t maxabs = 0;
    for (uint16_t i = 0; i < len; i++) {
        int32_t d = (int32_t)x[i] - mean;
        if (d < 0) d = -d;
        if (d > maxabs) maxabs = d;
    }
    if (maxabs == 0) {
        for (uint16_t i = 0; i < len; i++) out[i] = 0;
        return;
    }
    for (uint16_t i = 0; i < len; i++) {
        int32_t q = tc_div_round(((int32_t)x[i] - mean) * 127, maxabs);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out[i] = (int8_t)q;
    }
}

/* ================================================================ 各层运算 */
static inline int32_t conv_acc(const tc_layer_t *L, const int8_t *in, int oc, int t, int32_t bias)
{
    const int cg = L->in_ch / L->groups;
    const int og = L->out_ch / L->groups;
    const int K = L->kernel;
    const int ic0 = (oc / og) * cg;
    const int8_t *wrow = L->w + oc * cg * K;
    const int start = t * L->stride - L->pad;
    int k0 = (start < 0) ? -start : 0;
    int k1 = L->in_len - start;
    if (k1 > K) k1 = K;
    int32_t acc = bias;
    for (int c = 0; c < cg; c++) {
        const int8_t *xi = in + (ic0 + c) * L->in_len;
        const int8_t *wk = wrow + c * K;
        for (int k = k0; k < k1; k++) acc += (int32_t)xi[start + k] * wk[k];
    }
    return acc;
}

static void run_conv(const tc_layer_t *L, const int8_t *in, int8_t *out)
{
    const int cal = is_calib_layer(L);
    for (int oc = 0; oc < L->out_ch; oc++) {
        const int32_t bias = cal ? s_bias[L->ch_off + oc] : L->b[oc];
        const int32_t mult = cal ? s_mult[L->ch_off + oc] : L->mult[oc];
        const int8_t shift = cal ? s_shift[L->ch_off + oc] : L->shift[oc];
        int8_t *dst = out + oc * L->out_len;
        for (int t = 0; t < L->out_len; t++)
            dst[t] = requant(conv_acc(L, in, oc, t, bias), mult, shift, L->relu);
    }
}

static void run_fc(const tc_layer_t *L, const int8_t *in, int8_t *out, int32_t *logits)
{
    const int n_in = L->in_ch;                    /* 已展平的输入长度 */
    for (int o = 0; o < L->out_ch; o++) {
        const int8_t *w = L->w + o * n_in;
        int32_t acc = L->b[o];
        for (int i = 0; i < n_in; i++) acc += (int32_t)in[i] * w[i];
        if (L->last) logits[o] = acc;
        else out[o] = requant(acc, L->mult[o], L->shift[o], L->relu);
    }
}

static void run_maxpool(const tc_layer_t *L, const int8_t *in, int8_t *out)
{
    const int k = L->kernel;
    for (int c = 0; c < L->in_ch; c++) {
        const int8_t *src = in + c * L->in_len;
        int8_t *dst = out + c * L->out_len;
        for (int t = 0; t < L->out_len; t++) {
            int8_t m = src[t * k];
            for (int j = 1; j < k; j++)
                if (src[t * k + j] > m) m = src[t * k + j];
            dst[t] = m;
        }
    }
}

static void run_gap(const tc_layer_t *L, const int8_t *in, int8_t *out)
{
    for (int c = 0; c < L->in_ch; c++) {
        int32_t s = 0;
        const int8_t *src = in + c * L->in_len;
        for (int t = 0; t < L->in_len; t++) s += src[t];
        out[c] = (int8_t)tc_div_round(s, L->in_len);
    }
}

/* 执行第 0 .. n-1 层，返回第 n 层的输入指针 */
static const int8_t *run_layers(const int8_t *in, int n, int32_t *logits)
{
    const int8_t *cur = in;
    for (int i = 0; i < n; i++) {
        const tc_layer_t *L = &tc_layers[i];
        int8_t *nxt = (cur == s_buf_a) ? s_buf_b : s_buf_a;
        switch (L->type) {
        case TC_CONV:    run_conv(L, cur, nxt); break;
        case TC_FC:      run_fc(L, cur, nxt, logits); break;
        case TC_MAXPOOL: run_maxpool(L, cur, nxt); break;
        case TC_GAP:     run_gap(L, cur, nxt); break;
        case TC_FLATTEN: nxt = (int8_t *)cur; break;   /* [C][L] 已连续存放 */
        default: return 0;
        }
        cur = nxt;
    }
    return cur;
}

int tc_infer(const int8_t *in, int32_t *logits)
{
    if (!run_layers(in, TC_NUM_LAYERS, logits)) return -1;
    int best = 0;
    for (int c = 1; c < TC_NUM_CLASSES; c++)
        if (logits[c] > logits[best]) best = c;
    return best;
}

/* ================================================================ 片上重校准 */
void tc_reset_params(void)
{
    for (int i = 0; i < TC_NUM_LAYERS; i++) {
        const tc_layer_t *L = &tc_layers[i];
        if (!is_calib_layer(L)) continue;
        for (int c = 0; c < L->out_ch; c++) {
            s_bias[L->ch_off + c] = L->b[c];
            s_mult[L->ch_off + c] = L->mult[c];
            s_shift[L->ch_off + c] = L->shift[c];
        }
    }
}

int tc_num_calib_layers(void) { return TC_NUM_CALIB; }

void tc_calib_begin(void)
{
    for (int c = 0; c < TC_MAX_CALIB_CH; c++) { s_sum1[c] = 0; s_sum2[c] = 0; }
    s_cnt = 0;
}

int tc_calib_accumulate(int k, const int8_t *in)
{
    if (k < 0 || k >= TC_NUM_CALIB) return -1;
    const int li = tc_calib_layers[k];
    const tc_layer_t *L = &tc_layers[li];
    const int8_t *x = run_layers(in, li, 0);
    if (!x) return -2;
    for (int oc = 0; oc < L->out_ch; oc++) {
        const int32_t bias = s_bias[L->ch_off + oc];
        int64_t a1 = 0, a2 = 0;
        for (int t = 0; t < L->out_len; t++) {
            const int64_t acc = conv_acc(L, x, oc, t, bias);
            a1 += acc;
            a2 += acc * acc;
        }
        s_sum1[oc] += a1;
        s_sum2[oc] += a2;
    }
    s_cnt += L->out_len;
    return 0;
}

int tc_calib_apply(int k, int anchor, int scale)
{
    if (k < 0 || k >= TC_NUM_CALIB || s_cnt == 0) return -1;
    const tc_layer_t *L = &tc_layers[tc_calib_layers[k]];
    const int32_t *mu_s_arr = (anchor == TC_ANCHOR_ALL) ? L->mu_a : L->mu_h;
    const int32_t *sig_s_arr = (anchor == TC_ANCHOR_ALL) ? L->sig_a : L->sig_h;
    if (!mu_s_arr || !sig_s_arr) return -2;                 /* 模型未导出源域统计量 */
    const int64_t n = (int64_t)s_cnt;
    for (int c = 0; c < L->out_ch; c++) {
        const int64_t mu_t = div_round64(s_sum1[c], n);
        int64_t var = div_round64(s_sum2[c], n) - mu_t * mu_t;
        if (var < 0) var = 0;
        const int64_t sig_t = (int64_t)isqrt64((uint64_t)var);
        const int64_t mu_s = mu_s_arr[c], sig_s = sig_s_arr[c];

        int64_t aq = CAL_ONE;
        if (scale && sig_t > 0 && sig_s > 0) {
            aq = div_round64(sig_s * CAL_ONE, sig_t);
            if (aq < CAL_AQ_MIN) aq = CAL_AQ_MIN;
            if (aq > CAL_AQ_MAX) aq = CAL_AQ_MAX;
        }
        const int j = L->ch_off + c;
        int64_t b = (int64_t)s_bias[j] - mu_t + div_round64(mu_s * CAL_ONE, aq);
        if (b < INT32_MIN) b = INT32_MIN;
        if (b > INT32_MAX) b = INT32_MAX;
        int64_t m = ((int64_t)s_mult[j] * aq + (CAL_ONE >> 1)) >> CAL_Q;
        int s = s_shift[j];
        while (m >= ((int64_t)1 << 31)) { m = (m + 1) >> 1; s--; }
        while (m < ((int64_t)1 << 30)) { m <<= 1; s++; }
        if (s < 1) s = 1;
        if (s > 62) s = 62;
        s_bias[j] = (int32_t)b;
        s_mult[j] = (int32_t)m;
        s_shift[j] = (int8_t)s;
    }
    return 0;
}

int tc_get_params(int k, const int32_t **b, const int32_t **mult, const int8_t **shift)
{
    if (k < 0 || k >= TC_NUM_CALIB) return -1;
    const tc_layer_t *L = &tc_layers[tc_calib_layers[k]];
    *b = &s_bias[L->ch_off];
    *mult = &s_mult[L->ch_off];
    *shift = &s_shift[L->ch_off];
    return L->out_ch;
}
