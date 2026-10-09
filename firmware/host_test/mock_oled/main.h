/* 电脑端测试 ssd1306.c 用的最小 HAL（模拟 I2C 与 SSD1306 显存） */
#ifndef MOCK_OLED_MAIN_H
#define MOCK_OLED_MAIN_H
#include <stdint.h>
#define HAL_I2C_MODULE_ENABLED
typedef enum { HAL_OK = 0, HAL_ERROR, HAL_BUSY, HAL_TIMEOUT } HAL_StatusTypeDef;
typedef struct { uint32_t dummy; } I2C_HandleTypeDef;
#define I2C_MEMADD_SIZE_8BIT 1U
HAL_StatusTypeDef HAL_I2C_IsDeviceReady(I2C_HandleTypeDef *h, uint16_t addr, uint32_t trials, uint32_t t);
HAL_StatusTypeDef HAL_I2C_Mem_Write(I2C_HandleTypeDef *h, uint16_t addr, uint16_t mem, uint16_t msize,
                                    uint8_t *d, uint16_t n, uint32_t t);
#endif
