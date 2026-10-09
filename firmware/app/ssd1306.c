/*
 * ssd1306.c —— SSD1306 OLED 驱动实现（I2C 页寻址模式，不开整屏缓冲，只用 128 字节行缓冲）
 */
#include "main.h"          /* CubeMX 生成，包含 HAL 与 stm32f1xx_hal_conf.h */
#include "ssd1306.h"

#if defined(HAL_I2C_MODULE_ENABLED)

#include <string.h>
#include "font5x8.h"

extern I2C_HandleTypeDef hi2c1;

#define OLED_W        128
#define OLED_TIMEOUT  20U           /* 每次 I2C 传输的超时（毫秒） */

static uint16_t s_addr = 0;          /* 8 位写地址（0x78 或 0x7A），0 表示没有屏幕 */
static uint8_t  s_buf[OLED_W];       /* 一行（一页）的显示数据 */

static int oled_cmd(const uint8_t *c, uint16_t n)
{
    return HAL_I2C_Mem_Write(&hi2c1, s_addr, 0x00, I2C_MEMADD_SIZE_8BIT, (uint8_t *)c, n,
                             OLED_TIMEOUT) == HAL_OK ? 0 : -1;
}

static void oled_write_page(uint8_t page)
{
    const uint8_t pos[3] = { (uint8_t)(0xB0 | (page & 7)), 0x00, 0x10 };   /* 页地址、列地址 0 */
    if (oled_cmd(pos, 3) != 0) return;
    HAL_I2C_Mem_Write(&hi2c1, s_addr, 0x40, I2C_MEMADD_SIZE_8BIT, s_buf, OLED_W, OLED_TIMEOUT);
}

int oled_init(void)
{
    static const uint8_t init_seq[] = {
        0xAE,             /* 关显示 */
        0xD5, 0x80,       /* 时钟分频 */
        0xA8, 0x3F,       /* 多路复用率 64 */
        0xD3, 0x00,       /* 显示偏移 0 */
        0x40,             /* 起始行 0 */
        0x8D, 0x14,       /* 打开电荷泵（3.3V 供电模块必需） */
        0x20, 0x02,       /* 页寻址模式 */
        0xA1,             /* 列地址重映射 */
        0xC8,             /* COM 扫描方向反转（正常朝向） */
        0xDA, 0x12,       /* COM 引脚配置 */
        0x81, 0xCF,       /* 对比度 */
        0xD9, 0xF1,       /* 预充电周期 */
        0xDB, 0x40,       /* VCOMH */
        0xA4,             /* 显示 RAM 内容 */
        0xA6,             /* 正常显示（非反色） */
        0x2E,             /* 关闭滚动 */
        0xAF              /* 开显示 */
    };
    static const uint16_t cand[2] = { 0x3C << 1, 0x3D << 1 };   /* 常见的两个 I2C 地址 */
    s_addr = 0;
    for (int i = 0; i < 2; i++) {
        if (HAL_I2C_IsDeviceReady(&hi2c1, cand[i], 2, 5) == HAL_OK) { s_addr = cand[i]; break; }
    }
    if (s_addr == 0) return -1;
    if (oled_cmd(init_seq, sizeof(init_seq)) != 0) { s_addr = 0; return -1; }
    oled_clear();
    return 0;
}

int oled_ok(void) { return s_addr != 0; }

void oled_clear(void)
{
    if (!s_addr) return;
    memset(s_buf, 0, sizeof(s_buf));
    for (uint8_t p = 0; p < OLED_LINES; p++) oled_write_page(p);
}

static const uint8_t *glyph(char c)
{
    if (c < FONT5X8_FIRST || c > FONT5X8_LAST) c = '?';
    return font5x8[c - FONT5X8_FIRST];
}

void oled_text(uint8_t line, const char *s)
{
    if (!s_addr || line >= OLED_LINES) return;
    memset(s_buf, 0, sizeof(s_buf));
    for (int i = 0; s && s[i] && i < OLED_COLS; i++) memcpy(&s_buf[i * FONT5X8_W], glyph(s[i]), FONT5X8_W);
    oled_write_page(line);
}

/* 把一个字节的低 4 位（或高 4 位）纵向放大 2 倍：每个像素变成上下两个像素 */
static uint8_t stretch4(uint8_t nib)
{
    uint8_t r = 0;
    for (int b = 0; b < 4; b++) if (nib & (1u << b)) r |= (uint8_t)(3u << (2 * b));
    return r;
}

void oled_text_big(uint8_t line, const char *s)
{
    if (!s_addr || line + 1 >= OLED_LINES) return;
    for (int half = 0; half < 2; half++) {                  /* 0：上半页，1：下半页 */
        memset(s_buf, 0, sizeof(s_buf));
        for (int i = 0; s && s[i] && i < OLED_BIG_COLS; i++) {
            const uint8_t *g = glyph(s[i]);
            for (int x = 0; x < FONT5X8_W; x++) {
                uint8_t v = stretch4(half ? (uint8_t)(g[x] >> 4) : (uint8_t)(g[x] & 0x0F));
                s_buf[i * 2 * FONT5X8_W + 2 * x] = v;        /* 横向也放大 2 倍 */
                s_buf[i * 2 * FONT5X8_W + 2 * x + 1] = v;
            }
        }
        oled_write_page((uint8_t)(line + half));
    }
}

#else   /* 没有启用 I2C：提供空函数，其余功能照常工作 */

int  oled_init(void) { return -1; }
int  oled_ok(void) { return 0; }
void oled_clear(void) { }
void oled_text(uint8_t line, const char *s) { (void)line; (void)s; }
void oled_text_big(uint8_t line, const char *s) { (void)line; (void)s; }

#endif
