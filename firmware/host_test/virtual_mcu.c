/*
 * 虚拟单片机：在电脑上运行真实的固件代码（app.c / tinycnn.c / adxl345.c），
 * 串口换成标准输入输出，ADXL345 换成模拟信号。用于没有硬件时测试电脑端全部串口脚本。
 * 仅支持 Linux / macOS / WSL。编译：python build_virtual_mcu.py --model-dir export/tiny
 * 使用：各脚本的 --port 写成 virtual:export/tiny
 */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include "main.h"
#include "app.h"

GPIO_TypeDef mock_gpioa; CoreDebug_Type mock_cd; DWT_Type mock_dwt;
uint32_t SystemCoreClock = 72000000;
static USART_TypeDef usart1;
UART_HandleTypeDef huart1 = { &usart1 };
SPI_HandleTypeDef hspi1;

DWT_Type *mock_dwt_tick(void) { mock_dwt.CYCCNT += 50; return &mock_dwt; }
void HAL_Delay(uint32_t ms)
{
    struct timespec ts = { (time_t)(ms / 1000), (long)(ms % 1000) * 1000000L };
    nanosleep(&ts, 0);
}
uint32_t HAL_GetTick(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000 + ts.tv_nsec / 1000000);
}
void HAL_GPIO_WritePin(GPIO_TypeDef *p, uint16_t pin, GPIO_PinState s) { (void)p; (void)pin; (void)s; }

/* ---------------- 串口：标准输入输出 ---------------- */
static int read_full(uint8_t *d, size_t n)
{
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(STDIN_FILENO, d + got, n - got);
        if (r <= 0) exit(0);                         /* 电脑端关闭，退出 */
        got += (size_t)r;
    }
    return 0;
}

HAL_StatusTypeDef HAL_UART_Receive(UART_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t)
{
    (void)h; (void)t;
    read_full(d, n);
    return HAL_OK;
}

HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t)
{
    (void)h; (void)t;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(STDOUT_FILENO, d + off, n - off);
        if (w <= 0) exit(0);
        off += (size_t)w;
    }
    return HAL_OK;
}

int mock_uart_flag(UART_HandleTypeDef *h, uint32_t f)
{
    (void)h;
    if (f != UART_FLAG_RXNE) return 0;
    struct pollfd p = { STDIN_FILENO, POLLIN, 0 };
    if (poll(&p, 1, 0) > 0 && (p.revents & POLLIN)) {
        uint8_t b;
        read_full(&b, 1);
        usart1.DR = b;
        return 1;
    }
    return 0;
}

/* ---------------- 模拟 ADXL345：40 Hz 转频 + 谐波 + 噪声 ---------------- */
static uint8_t spi_reg;
static uint32_t sample_idx;
HAL_StatusTypeDef HAL_SPI_Transmit(SPI_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t)
{
    (void)h; (void)t; (void)n; spi_reg = d[0] & 0x3F; return HAL_OK;
}
HAL_StatusTypeDef HAL_SPI_Receive(SPI_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t)
{
    (void)h; (void)t; (void)n;
    if (spi_reg == 0x00) d[0] = 0xE5;
    else if (spi_reg == 0x39) d[0] = 16;
    else if (spi_reg == 0x32) {
        const double tt = sample_idx++ / 3200.0;
        for (int a = 0; a < 3; a++) {
            double v = (a == 2 ? 300.0 : 80.0) * sin(2 * 3.14159265358979 * 40.0 * tt + a)
                     + 60.0 * sin(2 * 3.14159265358979 * 120.0 * tt) + (rand() % 41 - 20);
            int16_t s = (int16_t)v;
            d[2 * a] = (uint8_t)s;
            d[2 * a + 1] = (uint8_t)((uint16_t)s >> 8);
        }
    }
    return HAL_OK;
}

int main(void)
{
    app_init();
    for (;;) app_loop();
}
