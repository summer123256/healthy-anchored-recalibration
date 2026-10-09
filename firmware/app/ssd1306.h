/*
 * ssd1306.h —— 0.96 寸 128×64 SSD1306 OLED 显示驱动（I2C，HAL 库）
 *
 * 接线：VCC→3.3V，GND→GND，SCL→PB6，SDA→PB7（I2C1）
 * CubeMX：Connectivity → I2C1 → I2C（Fast Mode 400 kHz 或 Standard Mode 100 kHz 均可）
 *
 * 屏幕按“页”划分为 8 行（第 0–7 行），每行 8 像素高，普通字每行 25 个字符。
 * 大字（2 倍）占两行，每行 12 个字符。
 *
 * 如果 CubeMX 里没有启用 I2C（没有 HAL_I2C_MODULE_ENABLED），或者上电时检测不到屏幕，
 * 所有函数都直接返回，不影响推理、校准和串口通信。
 */
#ifndef SSD1306_H
#define SSD1306_H

#include <stdint.h>

#define OLED_LINES      8
#define OLED_COLS       25      /* 普通字每行字符数（5 像素宽） */
#define OLED_BIG_COLS   12      /* 大字每行字符数（10 像素宽） */

int  oled_init(void);                                  /* 0：检测到屏幕并初始化成功；-1：没有屏幕 */
int  oled_ok(void);                                    /* 1：屏幕可用 */
void oled_clear(void);
void oled_text(uint8_t line, const char *s);           /* 在第 line 行写字，行内其余部分清空 */
void oled_text_big(uint8_t line, const char *s);       /* 2 倍大字，占用 line 与 line+1 两行 */

#endif
