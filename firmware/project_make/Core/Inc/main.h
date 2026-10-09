/*
 * main.h —— 与 STM32CubeMX 生成的 main.h 等价的参考版本
 *
 * 引脚标签与《动手实验方案》4.4 节、5.2 节的 CubeMX 设置一致：
 *   PA4  GPIO_Output  ADXL_CS（ADXL345 片选）
 *   PC13 GPIO_Output  LED（Blue Pill 板载 LED，低电平点亮）
 *   PB0  GPIO_Input   KEY（按键，上拉）
 * 用 CubeMX 生成工程时，这些宏由 CubeMX 根据 User Label 自动生成，不需要这个文件。
 */
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stm32f1xx_hal.h"

void Error_Handler(void);

#define LED_Pin            GPIO_PIN_13
#define LED_GPIO_Port      GPIOC
#define ADXL_CS_Pin        GPIO_PIN_4
#define ADXL_CS_GPIO_Port  GPIOA
#define KEY_Pin            GPIO_PIN_0
#define KEY_GPIO_Port      GPIOB

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
