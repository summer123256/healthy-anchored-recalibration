/*
 * test_oled.c —— 在电脑上测试 ssd1306.c：模拟 SSD1306 的页寻址显存，把屏幕内容打印成字符画。
 * 编译：gcc -I mock_oled -I ../app test_oled.c ../app/ssd1306.c -o test_oled
 */
#include <stdio.h>
#include <string.h>
#include "main.h"
#include "ssd1306.h"

I2C_HandleTypeDef hi2c1;
static uint8_t gddram[8][128];
static int page, col, present_addr = 0x3C << 1, inited = 0;

HAL_StatusTypeDef HAL_I2C_IsDeviceReady(I2C_HandleTypeDef *h, uint16_t addr, uint32_t trials, uint32_t t)
{
    (void)h; (void)trials; (void)t;
    return addr == present_addr ? HAL_OK : HAL_ERROR;
}

HAL_StatusTypeDef HAL_I2C_Mem_Write(I2C_HandleTypeDef *h, uint16_t addr, uint16_t mem, uint16_t msize,
                                    uint8_t *d, uint16_t n, uint32_t t)
{
    (void)h; (void)msize; (void)t;
    if (addr != present_addr) return HAL_ERROR;
    if (mem == 0x00) {                              /* 命令 */
        for (int i = 0; i < n; i++) {
            if (d[i] >= 0xB0 && d[i] <= 0xB7) page = d[i] - 0xB0;
            else if (d[i] <= 0x0F) col = (col & 0xF0) | d[i];
            else if (d[i] >= 0x10 && d[i] <= 0x1F) col = (col & 0x0F) | ((d[i] & 0x0F) << 4);
            else if (d[i] == 0xAF) inited = 1;
        }
    } else if (mem == 0x40) {                       /* 显示数据 */
        for (int i = 0; i < n && col < 128; i++) gddram[page][col++] = d[i];
    }
    return HAL_OK;
}

static int dump(int p0, int p1)
{
    int on = 0;
    for (int p = p0; p <= p1; p++)
        for (int bit = 0; bit < 8; bit++) {
            char row[129];
            for (int c = 0; c < 128; c++) {
                int v = (gddram[p][c] >> bit) & 1;
                on += v;
                row[c] = v ? '#' : '.';
            }
            row[128] = 0;
            printf("%s\n", row);
        }
    return on;
}

int main(void)
{
    if (oled_init() != 0 || !inited) { printf("FAIL: init\n"); return 1; }
    oled_text(0, "TinyML FaultDx STM32F103");
    oled_text_big(2, "OR014");
    oled_text(4, "Infer   12.34 ms");
    printf("第 0 行（普通字）:\n");
    int a = dump(0, 0);
    printf("第 2-3 行（大字）:\n");
    int b = dump(2, 3);
    printf("第 4 行:\n");
    int c = dump(4, 4);
    memset(gddram, 0, sizeof(gddram));
    present_addr = 0x3D << 1;                         /* 另一种常见地址 */
    if (oled_init() != 0) { printf("FAIL: addr 0x3D\n"); return 1; }
    present_addr = 0x11;                              /* 没有屏幕 */
    if (oled_init() == 0 || oled_ok()) { printf("FAIL: absent\n"); return 1; }
    oled_text(0, "no crash");                         /* 没有屏幕时调用也不能出错 */
    if (a == 0 || b == 0 || c == 0) { printf("FAIL: empty\n"); return 1; }
    printf("OLED 驱动测试通过\n");
    return 0;
}
