/*
 * app.c —— 串口命令处理：硬件在环推理、数据采集、在线诊断、片上重校准；
 *          OLED 显示、按键触发校准与独立演示模式（v3 新增）
 * 通信协议见 python/serial_proto.py
 *
 * OLED 与按键都是可选的：
 *   - CubeMX 没启用 I2C1，或上电时检测不到屏幕 → 不显示，其余功能不变；
 *   - CubeMX 没给 PB0 设置 User Label “KEY” → 没有按键功能，其余功能不变。
 * 与电脑通信时，OLED 只在单片机发送回复“之前”刷新（此时电脑在等待、不会发数据），
 * 因此不会丢串口数据，也不影响时钟周期计数（计时只包含预处理和推理本身）。
 */
#include <string.h>
#include "main.h"
#include "app.h"
#include "adxl345.h"
#include "ssd1306.h"
#include "tinycnn.h"
#include "model_data.h"
#include "test_vectors.h"

extern UART_HandleTypeDef huart1;

enum {
    CMD_INFER = 0x01, CMD_STREAM = 0x02, CMD_STOP = 0x03, CMD_ONLINE = 0x04,
    CMD_SELFTEST = 0x05, CMD_INFO = 0x06, CMD_CALIBRATE = 0x07, CMD_CALIB_FEED = 0x08,
    CMD_CALIB_APPLY = 0x09, CMD_RESET_PARAMS = 0x0A, CMD_DUMP = 0x0B,
    RSP_INFER = 0x81, RSP_STREAM = 0x82, RSP_ONLINE = 0x84, RSP_SELFTEST = 0x85,
    RSP_INFO = 0x86, RSP_CALIBRATE = 0x87, RSP_CALIB_FEED = 0x88, RSP_CALIB_APPLY = 0x89,
    RSP_RESET_PARAMS = 0x8A, RSP_DUMP = 0x8B, RSP_ERROR = 0xEE
};
enum { ERR_CHECKSUM = 1, ERR_LENGTH = 2, ERR_UNKNOWN = 3, ERR_SENSOR = 4, ERR_CALIB = 5 };

#define STR(x) #x
#define XSTR(x) STR(x)
#define STREAM_N 32                                /* 每帧发送的三轴样本数 */
#define RX_MAX   (2 * TC_INPUT_LEN + 8)
#define TX_MAX   (16 + 9 * TC_MAX_CALIB_CH + 4 * TC_NUM_CLASSES + 6 * STREAM_N)

static union { uint8_t b[RX_MAX]; int32_t align; } s_rx;          /* 接收缓冲区 */
static int16_t s_raw[TC_INPUT_LEN];                                  /* int16 原始窗口 */
static int8_t  s_q[TC_INPUT_LEN];                                    /* 预处理后的 int8 输入 */
static int32_t s_logits[TC_NUM_CLASSES];
static union { uint8_t b[TX_MAX]; int32_t align; } s_tx;
static int     s_sensor_ok = 0;
static uint16_t s_feed_count = 0;                                    /* 当前层已累加的窗口数 */
static int     s_calibrated = 0;                                     /* 是否已做过片上校准 */
static int     s_demo = 0;                                           /* 1：独立演示模式 */
static uint32_t s_last_disp = 0;                                     /* 上次刷新 OLED 的时刻（ms） */

static void ui_status(void);
static void ui_msg(const char *m);
static void ui_result(int pred, uint32_t cyc_pre, uint32_t cyc_inf);
static int  key_event(void);

/* ---------------------------------------------------------------- 计时 */
static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

void app_delay_us(uint32_t us)
{
    const uint32_t start = DWT->CYCCNT;
    const uint32_t ticks = us * (SystemCoreClock / 1000000U);
    while ((DWT->CYCCNT - start) < ticks) { }
}

/* ---------------------------------------------------------------- 串口收发 */
static void put_u16(uint8_t *p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
static void put_u32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }

/* 小工具：拼接显示用的字符串（不用 printf，节省 Flash） */
static char *put_str(char *p, const char *s) { while (*s) *p++ = *s++; *p = 0; return p; }
static char *put_uint(char *p, uint32_t v)
{
    char t[10];
    int n = 0;
    do { t[n++] = (char)('0' + v % 10U); v /= 10U; } while (v);
    while (n) *p++ = t[--n];
    *p = 0;
    return p;
}
/* 时钟周期数 → “x.xx” 毫秒 */
static char *put_ms(char *p, uint32_t cycles)
{
    uint32_t us = cycles / (SystemCoreClock / 1000000U);
    p = put_uint(p, us / 1000U);
    uint32_t f = (us % 1000U) / 10U;
    *p++ = '.';
    *p++ = (char)('0' + f / 10U);
    *p++ = (char)('0' + f % 10U);
    *p = 0;
    return p;
}

static void send_frame(uint8_t cmd, const uint8_t *payload, uint16_t len)
{
    uint8_t hdr[5] = { 0xA5, 0x5A, cmd, (uint8_t)len, (uint8_t)(len >> 8) };
    uint8_t cs = (uint8_t)(cmd + (uint8_t)len + (uint8_t)(len >> 8));
    for (uint16_t i = 0; i < len; i++) cs = (uint8_t)(cs + payload[i]);
    HAL_UART_Transmit(&huart1, hdr, 5, 100);
    if (len) HAL_UART_Transmit(&huart1, (uint8_t *)payload, len, 1000);
    HAL_UART_Transmit(&huart1, &cs, 1, 100);
}

static void send_error(uint8_t code) { send_frame(RSP_ERROR, &code, 1); }

static int rx_byte(uint8_t *b, uint32_t timeout)
{
    return HAL_UART_Receive(&huart1, b, 1, timeout) == HAL_OK ? 0 : -1;
}

/* 接收一帧。返回 0 成功；-1 超时（含空闲）；-2 校验错；-3 长度超限
 * 等待第一个字节最多 APP_POLL_MS 毫秒，超时返回 -1，让 app_loop 有机会检查按键。 */
static int recv_frame(uint8_t *cmd, uint8_t *payload, uint16_t maxlen, uint16_t *len)
{
    uint8_t b, prev = 0, h[3];
    uint32_t to = APP_POLL_MS;
    for (;;) {                                   /* 找帧头 A5 5A */
        if (rx_byte(&b, to) != 0) return -1;
        if (prev == 0xA5 && b == 0x5A) break;
        prev = b;
        to = 100;                                /* 帧开始后，字节间最多等 100 ms */
    }
    if (HAL_UART_Receive(&huart1, h, 3, 100) != HAL_OK) return -1;
    *cmd = h[0];
    *len = (uint16_t)(h[1] | (h[2] << 8));
    if (*len > maxlen) return -3;
    if (*len && HAL_UART_Receive(&huart1, payload, *len, 1000) != HAL_OK) return -1;
    uint8_t cs;
    if (rx_byte(&cs, 100) != 0) return -1;
    uint8_t sum = (uint8_t)(h[0] + h[1] + h[2]);
    for (uint16_t i = 0; i < *len; i++) sum = (uint8_t)(sum + payload[i]);
    return (sum == cs) ? 0 : -2;
}

/* 非阻塞检查是否收到 STOP 帧（A5 5A 03 00 00 03） */
static int poll_stop(void)
{
    static const uint8_t pat[6] = { 0xA5, 0x5A, CMD_STOP, 0x00, 0x00, CMD_STOP };
    static uint8_t pos = 0;
    while (__HAL_UART_GET_FLAG(&huart1, UART_FLAG_RXNE) ||
           __HAL_UART_GET_FLAG(&huart1, UART_FLAG_ORE)) {
        uint8_t b = (uint8_t)(huart1.Instance->DR & 0xFF);   /* 读 DR 同时清除 RXNE/ORE */
        if (b == pat[pos]) pos++;
        else pos = (b == pat[0]) ? 1 : 0;
        if (pos == sizeof(pat)) { pos = 0; return 1; }
    }
    return 0;
}

/* ---------------------------------------------------------------- 采集 */
/* 演示模式下的中止条件：按键，或电脑发来了数据（电脑要接管） */
static int demo_abort(void)
{
    return key_event() || __HAL_UART_GET_FLAG(&huart1, UART_FLAG_RXNE);
}

/* 采集一个连续的 1024 点窗口到 s_raw；收到 STOP（演示模式下为按键）返回 1 */
static int acquire_window(void)
{
    int16_t xyz[3];
    adxl_fifo_flush();                           /* 保证窗口内样本连续 */
    uint16_t i = 0;
    while (i < TC_INPUT_LEN) {
        if (s_demo ? demo_abort() : poll_stop()) return 1;
        uint8_t e = adxl_fifo_entries();
        while (e-- && i < TC_INPUT_LEN) {
            adxl_read_xyz(xyz);
            s_raw[i++] = xyz[APP_AXIS];
        }
    }
    return 0;
}

static void do_stream(void)
{
    uint16_t seq = 0;
    uint8_t n = 0, ovr = 0;
    int16_t *data = (int16_t *)(void *)&s_tx.b[4];   /* 偏移 4 处存放样本（4 字节对齐） */
    adxl_fifo_flush();
    while (!poll_stop()) {
        uint8_t e = adxl_fifo_entries();
        if (e >= 32) ovr = 1;                       /* FIFO 已满，可能丢样本 */
        while (e--) {
            adxl_read_xyz(&data[3 * n]);
            if (++n == STREAM_N) {
                put_u16(&s_tx.b[0], seq++);
                s_tx.b[2] = n;
                s_tx.b[3] = ovr;
                send_frame(RSP_STREAM, s_tx.b, (uint16_t)(4 + 6 * n));
                n = 0;
                ovr = 0;
            }
        }
    }
}

/* ---------------------------------------------------------------- 推理 */
static void infer_and_reply(uint8_t rsp)
{
    uint32_t t0 = DWT->CYCCNT;
    tc_preprocess(s_raw, s_q, TC_INPUT_LEN);
    uint32_t t1 = DWT->CYCCNT;
    int pred = tc_infer(s_q, s_logits);
    uint32_t t2 = DWT->CYCCNT;

    /* 先刷新显示再回复：电脑收到回复前不会发下一帧，不会丢数据。
     * 在线诊断模式（RSP_ONLINE）不刷新，以免错过电脑发来的 STOP。 */
    if (rsp == RSP_INFER && oled_ok() && (HAL_GetTick() - s_last_disp) >= APP_DISP_MS) {
        ui_result(pred, t1 - t0, t2 - t1);
        s_last_disp = HAL_GetTick();
    }

    s_tx.b[0] = (uint8_t)pred;
    put_u32(&s_tx.b[1], t1 - t0);
    put_u32(&s_tx.b[5], t2 - t1);
    for (int c = 0; c < TC_NUM_CLASSES; c++) put_u32(&s_tx.b[9 + 4 * c], (uint32_t)s_logits[c]);
    send_frame(rsp, s_tx.b, (uint16_t)(9 + 4 * TC_NUM_CLASSES));

#ifdef LED_Pin      /* 板载 LED（Blue Pill 为 PC13，低电平点亮）：非正常类别时点亮 */
    HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, pred == 0 ? GPIO_PIN_SET : GPIO_PIN_RESET);
#endif
}

static void do_online(void)
{
    for (;;) {
        if (acquire_window()) return;
        infer_and_reply(RSP_ONLINE);
    }
}

/* 片上自检：用 Flash 中的测试向量核对推理结果。会先恢复出厂参数（清除校准）。 */
static void do_selftest(void)
{
    uint8_t pass = 0;
    uint32_t cyc = 0;
    tc_reset_params();
    tc_calib_begin();
    s_feed_count = 0;
    s_calibrated = 0;
    for (int i = 0; i < TV_COUNT; i++) {
        tc_preprocess(tv_input[i], s_q, TC_INPUT_LEN);
        uint32_t t0 = DWT->CYCCNT;
        tc_infer(s_q, s_logits);
        cyc = DWT->CYCCNT - t0;
        pass += (memcmp(s_logits, tv_logits[i], sizeof(s_logits)) == 0);
    }
    ui_status();
    ui_msg(pass == TV_COUNT ? "Selftest: PASS" : "Selftest: FAIL");
    s_tx.b[0] = pass;
    s_tx.b[1] = (uint8_t)TV_COUNT;
    put_u32(&s_tx.b[2], cyc);
    send_frame(RSP_SELFTEST, s_tx.b, 6);
}

static void do_info(void)
{
    static const char info[] = "model=" TC_MODEL_NAME " input=" XSTR(TC_INPUT_LEN)
                               " classes=" XSTR(TC_NUM_CLASSES) " calib_layers=" XSTR(TC_NUM_CALIB)
                               " stats=" XSTR(TC_HAS_STATS) " fw=stm32f103-tinycnn-v3";
    send_frame(RSP_INFO, (const uint8_t *)info, (uint16_t)(sizeof(info) - 1));
}

/* ---------------------------------------------------------------- 片上重校准 */
/* 电脑发送一个窗口，累加到第 k 个可校准层（硬件在环校准，用于公开数据集） */
static void do_calib_feed(const uint8_t *pl, uint16_t len)
{
    if (len != 1 + 2 * TC_INPUT_LEN) { send_error(ERR_LENGTH); return; }
    memcpy(s_raw, pl + 1, 2 * TC_INPUT_LEN);
    uint32_t t0 = DWT->CYCCNT;
    tc_preprocess(s_raw, s_q, TC_INPUT_LEN);
    int r = tc_calib_accumulate(pl[0], s_q);            /* 前向到第 k 层并累加 Σacc、Σacc² */
    uint32_t cyc = DWT->CYCCNT - t0;                    /* 预处理 + 前向 + 累加 的周期数 */
    if (r == 0) s_feed_count++;
    s_tx.b[0] = (uint8_t)(int8_t)r;
    put_u16(&s_tx.b[1], s_feed_count);
    put_u32(&s_tx.b[3], cyc);
    send_frame(RSP_CALIB_FEED, s_tx.b, 7);
}

/* 用已累加的统计量更新第 k 层，然后清零累加器 */
static void do_calib_apply(const uint8_t *pl, uint16_t len)
{
    if (len != 3) { send_error(ERR_LENGTH); return; }
    uint32_t t0 = DWT->CYCCNT;
    int r = tc_calib_apply(pl[0], pl[1], pl[2]);
    uint32_t cyc = DWT->CYCCNT - t0;
    tc_calib_begin();
    s_feed_count = 0;
    if (r == 0) s_calibrated = 1;
    if (oled_ok()) {
        char m[OLED_COLS + 1];
        char *p = put_uint(put_str(m, "PC calib layer "), (uint32_t)pl[0] + 1U);
        put_str(p, r == 0 ? " ok" : " err");
        ui_status();
        ui_msg(m);
    }
    s_tx.b[0] = (uint8_t)(int8_t)r;
    put_u32(&s_tx.b[1], cyc);
    send_frame(RSP_CALIB_APPLY, s_tx.b, 5);
}

/* 单片机自主校准核心：设备健康运行时，逐层采集 n 个窗口并更新参数。
 * show：0 不显示；1 每个窗口显示进度（演示模式）；2 每层显示一次（电脑触发）。
 * 返回 0 成功；-1 累加失败；-2 更新失败；-3 被中止。 */
static int8_t calibrate_run(int n, int anchor, int scale, int layers,
                            uint16_t *used, uint32_t *compute, int show)
{
    int8_t status = 0;
    *used = 0;
    *compute = 0;
    for (int k = 0; k < layers && status == 0; k++) {
        tc_calib_begin();
        for (int i = 0; i < n; i++) {
            if (show == 1 || (show == 2 && i == 0)) {
                char m[OLED_COLS + 1];
                char *p = put_uint(put_str(m, "Calib L"), (uint32_t)k + 1U);
                p = put_uint(put_str(p, "/"), (uint32_t)layers);
                p = put_uint(put_str(p, "  win "), (uint32_t)i + 1U);
                put_uint(put_str(p, "/"), (uint32_t)n);
                ui_msg(m);
            }
            if (acquire_window()) { status = -3; break; }       /* 收到 STOP 或按键，中止 */
            uint32_t t0 = DWT->CYCCNT;
            tc_preprocess(s_raw, s_q, TC_INPUT_LEN);
            if (tc_calib_accumulate(k, s_q) != 0) { status = -1; break; }
            *compute += DWT->CYCCNT - t0;
            (*used)++;
        }
        if (status == 0) {
            uint32_t t0 = DWT->CYCCNT;
            if (tc_calib_apply(k, anchor, scale) != 0) status = -2;
            *compute += DWT->CYCCNT - t0;
        }
    }
    tc_calib_begin();
    s_feed_count = 0;
    return status;
}

/* 电脑触发的自主校准（device_calibrate.py） */
static void do_calibrate(const uint8_t *pl, uint16_t len)
{
    if (len != 4) { send_error(ERR_LENGTH); return; }
    if (!s_sensor_ok) { send_error(ERR_SENSOR); return; }
    const int n = pl[0] ? pl[0] : 32;
    const int anchor = pl[1], scale = pl[2];
    int layers = pl[3] ? pl[3] : TC_NUM_CALIB;
    if (layers > TC_NUM_CALIB) layers = TC_NUM_CALIB;
    uint16_t used;
    uint32_t compute;
    tc_reset_params();                              /* 每次校准都从出厂参数开始，重复校准不会累积 */
    s_calibrated = 0;
    const uint32_t ms0 = HAL_GetTick();
    int8_t status = calibrate_run(n, anchor, scale, layers, &used, &compute, oled_ok() ? 2 : 0);
    if (status != 0) tc_reset_params();             /* 中止或失败：恢复出厂参数，避免只校准了一半 */
    const uint32_t ms = HAL_GetTick() - ms0;
    if (status == 0) s_calibrated = 1;
    ui_status();
    ui_msg(status == 0 ? "Calibration done" : "Calibration failed");
    s_tx.b[0] = (uint8_t)status;
    put_u16(&s_tx.b[1], used);
    put_u32(&s_tx.b[3], ms);
    put_u32(&s_tx.b[7], compute);
    send_frame(RSP_CALIBRATE, s_tx.b, 11);
}

/* 读取第 k 层当前参数，用于与 Python 逐位比对 */
static void do_dump(const uint8_t *pl, uint16_t len)
{
    if (len != 1) { send_error(ERR_LENGTH); return; }
    const int32_t *b, *m;
    const int8_t *s;
    int n = tc_get_params(pl[0], &b, &m, &s);
    if (n < 0) { send_error(ERR_CALIB); return; }
    s_tx.b[0] = (uint8_t)n;
    for (int c = 0; c < n; c++) {
        put_u32(&s_tx.b[1 + 4 * c], (uint32_t)b[c]);
        put_u32(&s_tx.b[1 + 4 * n + 4 * c], (uint32_t)m[c]);
        s_tx.b[1 + 8 * n + c] = (uint8_t)s[c];
    }
    send_frame(RSP_DUMP, s_tx.b, (uint16_t)(1 + 9 * n));
}

/* ---------------------------------------------------------------- OLED 显示 */
/* 屏幕布局：
 *   第 0 行  TinyML FaultDx
 *   第 1 行  模式（PC / DEMO）与是否已校准
 *   第 2–3 行  诊断类别（大字）
 *   第 4 行  推理耗时   第 5 行  预处理耗时
 *   第 6 行  状态信息   第 7 行  模型名 */
static void ui_status(void)
{
    if (!oled_ok()) return;
    char m[OLED_COLS + 1];
    oled_text(0, "TinyML FaultDx STM32F103");
    char *p = put_str(m, s_demo ? "Mode: DEMO " : "Mode: PC   ");
    put_str(p, s_calibrated ? "Calib: YES" : "Calib: NO");
    oled_text(1, m);
    oled_text(7, "model: " TC_MODEL_NAME);      /* 超出一行的部分不显示 */
}

static void ui_msg(const char *m)
{
    oled_text(6, m);
}

static void ui_result(int pred, uint32_t cyc_pre, uint32_t cyc_inf)
{
    if (!oled_ok()) return;
    char m[OLED_COLS + 1];
    oled_text_big(2, (pred >= 0 && pred < TC_NUM_CLASSES) ? tc_class_names[pred] : "?");
    put_str(put_ms(put_str(m, "Infer   "), cyc_inf), " ms");
    oled_text(4, m);
    put_str(put_ms(put_str(m, "Preproc "), cyc_pre), " ms");
    oled_text(5, m);
}

/* ---------------------------------------------------------------- 按键 */
#ifdef KEY_Pin           /* CubeMX 中把 PB0 设为 GPIO_Input 并标注 User Label “KEY” 后生效 */
static int key_down(void)
{
    return HAL_GPIO_ReadPin(KEY_GPIO_Port, KEY_Pin) == APP_KEY_ACTIVE;
}

/* 检测一次按键：按下并消抖 20 ms 后等待松开（最多 2 秒）。返回 1 表示有一次按键 */
static int key_event(void)
{
    if (!key_down()) return 0;
    HAL_Delay(20);
    if (!key_down()) return 0;
    const uint32_t t0 = HAL_GetTick();
    while (key_down() && (HAL_GetTick() - t0) < 2000U) { }
    return 1;
}
#else
static int key_event(void) { return 0; }
#endif

/* ---------------------------------------------------------------- 独立演示模式 */
/* 按一下按键进入：先用 ADXL345 采集的“正常运行”数据做片上校准（本文方法，健康锚点），
 * 然后连续采集、诊断并在 OLED 上显示结果。再按一下按键（或电脑发来数据）退出。
 * 注意：ADXL345 最高 3200 Hz，与公开数据集的采样率不同，这里显示的类别只用于演示流程，
 * 不代表诊断准确率。 */
static void run_demo(void)
{
    s_demo = 1;
    ui_status();
    if (!s_sensor_ok) {
        ui_msg("No ADXL345 sensor");
        HAL_Delay(1500);
        s_demo = 0;
        ui_status();
        return;
    }
    uint16_t used;
    uint32_t compute;
    tc_reset_params();
    s_calibrated = 0;
    int8_t st = calibrate_run(APP_DEMO_N, TC_ANCHOR_HEALTHY, 1,
                             APP_DEMO_LAYERS < TC_NUM_CALIB ? APP_DEMO_LAYERS : TC_NUM_CALIB,
                             &used, &compute, 1);
    if (st != 0) {                                  /* 中止或失败：恢复出厂参数，避免只校准了一半 */
        tc_reset_params();
        s_demo = 0;
        ui_status();
        ui_msg(st == -3 ? "Calibration aborted" : "Calibration error");
        return;
    }
    s_calibrated = 1;
    ui_status();
    ui_msg("Calibrated. Running");
    for (;;) {
        if (acquire_window()) break;
        uint32_t t0 = DWT->CYCCNT;
        tc_preprocess(s_raw, s_q, TC_INPUT_LEN);
        uint32_t t1 = DWT->CYCCNT;
        int pred = tc_infer(s_q, s_logits);
        uint32_t t2 = DWT->CYCCNT;
        ui_result(pred, t1 - t0, t2 - t1);
#ifdef LED_Pin
        HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, pred == 0 ? GPIO_PIN_SET : GPIO_PIN_RESET);
#endif
    }
    s_demo = 0;
    ui_status();
    ui_msg("Back to PC mode");
}

/* ---------------------------------------------------------------- 入口 */
void app_init(void)
{
    dwt_init();
    tc_reset_params();
    tc_calib_begin();
    s_sensor_ok = (adxl_init() == 0);
    if (oled_init() == 0) {
        ui_status();
        oled_text_big(2, "READY");
        ui_msg(s_sensor_ok ? "ADXL345: OK" : "ADXL345: not found");
    }
}

void app_loop(void)
{
    uint8_t cmd;
    uint16_t len;
    int r = recv_frame(&cmd, s_rx.b, RX_MAX, &len);
    if (r == -1) {                               /* 空闲：检查按键 */
        if (key_event()) run_demo();
        return;
    }
    if (r == -2) { send_error(ERR_CHECKSUM); return; }
    if (r == -3) { send_error(ERR_LENGTH); return; }

    switch (cmd) {
    case CMD_INFER:
        if (len != 2 * TC_INPUT_LEN) { send_error(ERR_LENGTH); break; }
        memcpy(s_raw, s_rx.b, 2 * TC_INPUT_LEN);       /* 小端 int16 */
        infer_and_reply(RSP_INFER);
        break;
    case CMD_STREAM:
        if (!s_sensor_ok) { send_error(ERR_SENSOR); break; }
        do_stream();
        break;
    case CMD_ONLINE:
        if (!s_sensor_ok) { send_error(ERR_SENSOR); break; }
        do_online();
        break;
    case CMD_SELFTEST:     do_selftest(); break;
    case CMD_INFO:         do_info(); break;
    case CMD_CALIBRATE:    do_calibrate(s_rx.b, len); break;
    case CMD_CALIB_FEED:   do_calib_feed(s_rx.b, len); break;
    case CMD_CALIB_APPLY:  do_calib_apply(s_rx.b, len); break;
    case CMD_RESET_PARAMS:
        tc_reset_params();
        tc_calib_begin();
        s_feed_count = 0;
        s_calibrated = 0;
        ui_status();
        ui_msg("Params reset");
        send_frame(RSP_RESET_PARAMS, 0, 0);
        break;
    case CMD_DUMP:         do_dump(s_rx.b, len); break;
    case CMD_STOP:         break;
    default:               send_error(ERR_UNKNOWN); break;
    }
}
