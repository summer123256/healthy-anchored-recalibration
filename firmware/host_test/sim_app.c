/*
 * 在电脑上仿真 app.c 的串口协议与数据流（模拟 HAL、UART、ADXL345）。
 * 验证：INFO / SELFTEST / INFER / STREAM+STOP / ONLINE+STOP，
 * 以及片上重校准相关的 RESET / CALIB_FEED / CALIB_APPLY / DUMP / CALIBRATE 命令。
 * 编译：见 run_sim.sh
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"
#include "app.h"
#include "model_data.h"

GPIO_TypeDef mock_gpioa; CoreDebug_Type mock_cd; DWT_Type mock_dwt;
uint32_t SystemCoreClock = 72000000;
DWT_Type *mock_dwt_tick(void) { mock_dwt.CYCCNT += 50; return &mock_dwt; }
static USART_TypeDef usart1;
UART_HandleTypeDef huart1 = { &usart1 };
SPI_HandleTypeDef hspi1;

/* ---------- 模拟串口 ---------- */
static uint8_t rxq[1 << 16]; static int rx_head, rx_tail;
static uint8_t txb[1 << 20]; static int tx_len;
static int stop_delay;                      /* 模拟：STOP 帧在若干次轮询之后才“到达” */

static void rx_push(const uint8_t *d, int n) { memcpy(rxq + rx_tail, d, n); rx_tail += n; }
int mock_uart_flag(UART_HandleTypeDef *h, uint32_t f)
{
    (void)h;
    if (stop_delay > 0) { stop_delay--; return 0; }
    if (f == UART_FLAG_RXNE && rx_head < rx_tail) { usart1.DR = rxq[rx_head++]; return 1; }
    return 0;
}
HAL_StatusTypeDef HAL_UART_Receive(UART_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t)
{
    (void)h; (void)t;
    if (rx_tail - rx_head < n) { printf("  [仿真] 接收队列已空，结束\n"); exit(0); }
    memcpy(d, rxq + rx_head, n); rx_head += n; return HAL_OK;
}
HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t)
{
    (void)h; (void)t; memcpy(txb + tx_len, d, n); tx_len += n; return HAL_OK;
}

/* ---------- 模拟 ADXL345 ---------- */
static uint8_t spi_reg; static int spi_n; static uint32_t sample_cnt;
void HAL_GPIO_WritePin(GPIO_TypeDef *p, uint16_t pin, GPIO_PinState s) { (void)p; (void)pin; (void)s; }
void HAL_Delay(uint32_t ms) { (void)ms; }
static uint32_t tick_ms;
uint32_t HAL_GetTick(void) { return tick_ms += 1; }
HAL_StatusTypeDef HAL_SPI_Transmit(SPI_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t)
{
    (void)h; (void)t; spi_reg = d[0] & 0x3F; spi_n = n; return HAL_OK;
}
HAL_StatusTypeDef HAL_SPI_Receive(SPI_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t)
{
    (void)h; (void)t;
    if (spi_reg == 0x00) d[0] = 0xE5;
    else if (spi_reg == 0x39) d[0] = 8;                  /* FIFO 中有 8 个样本 */
    else if (spi_reg == 0x32) {
        for (int a = 0; a < 3; a++) {
            int16_t v = (int16_t)(((int)(sample_cnt * 37 + a * 1000) % 2001) - 1000);
            d[2 * a] = (uint8_t)v; d[2 * a + 1] = (uint8_t)((uint16_t)v >> 8);
        }
        sample_cnt++;
    }
    (void)n; return HAL_OK;
}

/* ---------- 帧工具 ---------- */
static int make_frame(uint8_t *o, uint8_t cmd, const uint8_t *p, uint16_t n)
{
    o[0] = 0xA5; o[1] = 0x5A; o[2] = cmd; o[3] = (uint8_t)n; o[4] = (uint8_t)(n >> 8);
    uint8_t cs = (uint8_t)(cmd + o[3] + o[4]);
    for (int i = 0; i < n; i++) { o[5 + i] = p[i]; cs = (uint8_t)(cs + p[i]); }
    o[5 + n] = cs; return 6 + n;
}
static int parse_frames(int expect_cmd, int *count, uint8_t *first_payload)
{
    int i = 0, ok = 1; *count = 0;
    while (i < tx_len) {
        if (txb[i] != 0xA5 || txb[i + 1] != 0x5A) { printf("  帧头错误 @%d\n", i); return 0; }
        uint8_t cmd = txb[i + 2]; uint16_t n = (uint16_t)(txb[i + 3] | (txb[i + 4] << 8));
        uint8_t cs = (uint8_t)(cmd + txb[i + 3] + txb[i + 4]);
        for (int k = 0; k < n; k++) cs = (uint8_t)(cs + txb[i + 5 + k]);
        if (cs != txb[i + 5 + n]) { printf("  校验错误\n"); ok = 0; }
        if (cmd != expect_cmd) { printf("  命令不符 0x%02X\n", cmd); ok = 0; }
        if (*count == 0 && first_payload) memcpy(first_payload, txb + i + 5, n);
        (*count)++; i += 6 + n;
    }
    tx_len = 0; return ok;
}

int main(int argc, char **argv)
{
    uint8_t f[4096], pl[4096]; int n, cnt, fails = 0;
    int16_t x[TC_INPUT_LEN]; int32_t ref[TC_NUM_CLASSES];
    FILE *fp = fopen(argc > 1 ? argv[1] : "vectors.bin", "rb");
    if (!fp) { perror("vectors.bin"); return 2; }
    fseek(fp, 16, SEEK_SET);
    if (fread(x, 2, TC_INPUT_LEN, fp) != TC_INPUT_LEN || fread(ref, 4, TC_NUM_CLASSES, fp) != TC_NUM_CLASSES) return 2;
    fclose(fp);

    app_init();

    memset(pl, 0, sizeof(pl));
    n = make_frame(f, 0x06, NULL, 0); rx_push(f, n); app_loop();
    if (!parse_frames(0x86, &cnt, pl)) fails++;
    printf("INFO: %s\n", (char *)pl);

    n = make_frame(f, 0x05, NULL, 0); rx_push(f, n); app_loop();
    if (!parse_frames(0x85, &cnt, pl)) fails++;
    printf("SELFTEST: %u/%u 通过\n", pl[0], pl[1]); if (pl[0] != pl[1]) fails++;

    n = make_frame(f, 0x01, (uint8_t *)x, sizeof(x)); rx_push(f, n); app_loop();
    if (!parse_frames(0x81, &cnt, pl)) fails++;
    int same = memcmp(pl + 9, ref, sizeof(ref)) == 0;
    printf("INFER: pred=%u logits %s\n", pl[0], same ? "与 Python 一致" : "不一致!"); if (!same) fails++;

    n = make_frame(f, 0x02, NULL, 0); rx_push(f, n);
    n = make_frame(f, 0x03, NULL, 0); rx_push(f, n);
    sample_cnt = 0;
    stop_delay = 60;          /* STOP 帧在 60 次轮询之后才到达，期间应持续发送数据帧 */
    app_loop();
    if (!parse_frames(0x82, &cnt, pl)) fails++;
    printf("STREAM: 收到 %d 帧，首帧 n=%u，共采样 %u 组\n", cnt, pl[2], sample_cnt);
    if (cnt < 1) fails++;

    n = make_frame(f, 0x04, NULL, 0); rx_push(f, n);
    n = make_frame(f, 0x03, NULL, 0); rx_push(f, n);
    stop_delay = 1200;
    app_loop();
    if (!parse_frames(0x84, &cnt, pl)) fails++;
    printf("ONLINE: 收到 %d 个诊断结果，首个 pred=%u\n", cnt, pl[0]);
    if (cnt < 1) fails++;

    /* ---- 片上重校准：电脑逐窗口喂数据（硬件在环） ---- */
    stop_delay = 0;
    n = make_frame(f, 0x0A, NULL, 0); rx_push(f, n); app_loop();
    if (!parse_frames(0x8A, &cnt, pl)) fails++;
    for (int i = 0; i < 3; i++) {
        uint8_t fp[1 + sizeof(x)];
        fp[0] = 0;                                   /* 第 0 个可校准层 */
        memcpy(fp + 1, x, sizeof(x));
        n = make_frame(f, 0x08, fp, sizeof(fp)); rx_push(f, n); app_loop();
        if (!parse_frames(0x88, &cnt, pl) || (int8_t)pl[0] != 0) fails++;
    }
    printf("CALIB_FEED: 已累加 %u 个窗口\n", pl[1] | (pl[2] << 8));
    { uint8_t ap[3] = { 0, 0, 1 }; n = make_frame(f, 0x09, ap, 3); rx_push(f, n); app_loop(); }
    if (!parse_frames(0x89, &cnt, pl) || (int8_t)pl[0] != 0) fails++;
    printf("CALIB_APPLY: 状态 %d\n", (int8_t)pl[0]);
    { uint8_t dp[1] = { 0 }; n = make_frame(f, 0x0B, dp, 1); rx_push(f, n); app_loop(); }
    if (!parse_frames(0x8B, &cnt, pl)) fails++;
    printf("DUMP: 第 0 层 %u 个通道\n", pl[0]);

    /* ---- 片上重校准：单片机自主采集（模拟 ADXL345） ---- */
    { uint8_t cp[4] = { 2, 0, 1, 3 };            /* 每层 2 个窗口，健康锚点，校正尺度，前 3 层 */
      n = make_frame(f, 0x07, cp, 4); rx_push(f, n); app_loop(); }
    if (!parse_frames(0x87, &cnt, pl)) fails++;
    { int used = pl[1] | (pl[2] << 8);
      printf("CALIBRATE: 状态 %d，使用 %d 个窗口\n", (int8_t)pl[0], used);
      if ((int8_t)pl[0] != 0 || used != 6) fails++; }

    /* ---- 自检会恢复出厂参数，应再次全部通过 ---- */
    n = make_frame(f, 0x05, NULL, 0); rx_push(f, n); app_loop();
    if (!parse_frames(0x85, &cnt, pl) || pl[0] != pl[1]) fails++;
    printf("SELFTEST（恢复出厂参数后）: %u/%u 通过\n", pl[0], pl[1]);

    printf(fails ? "仿真失败 %d 项\n" : "协议仿真全部通过\n", fails);
    return fails ? 1 : 0;
}
