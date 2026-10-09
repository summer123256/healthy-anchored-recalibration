/*
 * adxl345.c —— ADXL345 SPI 驱动实现
 */
#include "main.h"          /* CubeMX 生成，包含 HAL 与引脚标签 */
#include "adxl345.h"
#include "app.h"

extern SPI_HandleTypeDef hspi1;

#ifndef ADXL_CS_Pin        /* 若 CubeMX 中没有把 PA4 标注为 ADXL_CS，则使用默认 PA4 */
#define ADXL_CS_GPIO_Port GPIOA
#define ADXL_CS_Pin       GPIO_PIN_4
#endif

#define CS_LOW()  HAL_GPIO_WritePin(ADXL_CS_GPIO_Port, ADXL_CS_Pin, GPIO_PIN_RESET)
#define CS_HIGH() HAL_GPIO_WritePin(ADXL_CS_GPIO_Port, ADXL_CS_Pin, GPIO_PIN_SET)

static void adxl_write(uint8_t reg, uint8_t val)
{
    uint8_t tx[2] = { (uint8_t)(reg & 0x3F), val };
    CS_LOW();
    HAL_SPI_Transmit(&hspi1, tx, 2, 10);
    CS_HIGH();
}

static void adxl_read(uint8_t reg, uint8_t *buf, uint16_t n)
{
    uint8_t addr = (uint8_t)(0x80 | (n > 1 ? 0x40 : 0x00) | (reg & 0x3F));
    CS_LOW();
    HAL_SPI_Transmit(&hspi1, &addr, 1, 10);
    HAL_SPI_Receive(&hspi1, buf, n, 10);
    CS_HIGH();
}

int adxl_init(void)
{
    uint8_t id = 0;
    CS_HIGH();
    HAL_Delay(5);
    adxl_read(ADXL_REG_DEVID, &id, 1);
    if (id != ADXL_DEVID_VALUE) return -1;
    adxl_write(ADXL_REG_POWER_CTL, 0x00);    /* 待机，便于修改配置 */
    adxl_write(ADXL_REG_DATA_FORMAT, 0x0B);  /* FULL_RES=1，±16 g，4 线 SPI */
    adxl_write(ADXL_REG_BW_RATE, 0x0F);      /* 输出数据率 3200 Hz */
    adxl_write(ADXL_REG_FIFO_CTL, 0x80);     /* FIFO 流模式 */
    adxl_write(ADXL_REG_POWER_CTL, 0x08);    /* 进入测量模式 */
    HAL_Delay(5);
    return 0;
}

uint8_t adxl_fifo_entries(void)
{
    uint8_t s = 0;
    adxl_read(ADXL_REG_FIFO_STATUS, &s, 1);
    return (uint8_t)(s & 0x3F);
}

void adxl_read_xyz(int16_t xyz[3])
{
    uint8_t b[6];
    adxl_read(ADXL_REG_DATAX0, b, 6);
    xyz[0] = (int16_t)((uint16_t)b[1] << 8 | b[0]);
    xyz[1] = (int16_t)((uint16_t)b[3] << 8 | b[2]);
    xyz[2] = (int16_t)((uint16_t)b[5] << 8 | b[4]);
    app_delay_us(5);   /* 数据手册：两次 FIFO 读取之间至少间隔 5 us */
}

void adxl_fifo_flush(void)
{
    int16_t tmp[3];
    uint8_t n = adxl_fifo_entries();
    while (n--) adxl_read_xyz(tmp);
}
