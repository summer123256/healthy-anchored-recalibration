/*
 * adxl345.h —— ADXL345 三轴加速度计 SPI 驱动（STM32 HAL）
 *
 * 配置：±16 g 全分辨率（3.9 mg/LSB）、输出数据率 3200 Hz、FIFO 流模式。
 * 接线（SPI1，4 线）：
 *   ADXL345  CS  -> PA4（GPIO 输出，CubeMX 中标签命名为 ADXL_CS）
 *            SCL -> PA5 (SPI1_SCK)
 *            SDO -> PA6 (SPI1_MISO)
 *            SDA -> PA7 (SPI1_MOSI)
 *            VCC -> 3.3V   GND -> GND
 * SPI 设置：模式 3（CPOL=High, CPHA=2 Edge），8 位，MSB 先行，分频 16（72MHz/16=4.5MHz，
 *           芯片上限 5 MHz；数据手册建议 3200 Hz 输出率时 SPI 时钟不低于 2 MHz）。
 */
#ifndef ADXL345_H
#define ADXL345_H

#include <stdint.h>

#define ADXL_REG_DEVID       0x00
#define ADXL_REG_BW_RATE     0x2C
#define ADXL_REG_POWER_CTL   0x2D
#define ADXL_REG_DATA_FORMAT 0x31
#define ADXL_REG_DATAX0      0x32
#define ADXL_REG_FIFO_CTL    0x38
#define ADXL_REG_FIFO_STATUS 0x39
#define ADXL_DEVID_VALUE     0xE5

int      adxl_init(void);                 /* 成功返回 0；读不到器件 ID 返回 -1 */
uint8_t  adxl_fifo_entries(void);         /* FIFO 中可读的样本数（0~32） */
void     adxl_read_xyz(int16_t xyz[3]);   /* 读取一组 X/Y/Z，并弹出 FIFO 一项 */
void     adxl_fifo_flush(void);           /* 丢弃 FIFO 中已有的数据 */

#endif
