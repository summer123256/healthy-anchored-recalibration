/*
 * stm32f1xx_hal_msp.c —— 外设引脚与时钟初始化（与 CubeMX 生成的内容等价）
 */
#include "main.h"

void HAL_MspInit(void)
{
    __HAL_RCC_AFIO_CLK_ENABLE();
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_AFIO_REMAP_SWJ_NOJTAG();     /* 调试口只用 SWD（Serial Wire），释放 JTAG 引脚 */
}

void HAL_I2C_MspInit(I2C_HandleTypeDef *hi2c)
{
    GPIO_InitTypeDef g = {0};
    if (hi2c->Instance == I2C1) {
        __HAL_RCC_GPIOB_CLK_ENABLE();
        g.Pin = GPIO_PIN_6 | GPIO_PIN_7;           /* PB6 SCL，PB7 SDA */
        g.Mode = GPIO_MODE_AF_OD;
        g.Speed = GPIO_SPEED_FREQ_HIGH;
        HAL_GPIO_Init(GPIOB, &g);
        __HAL_RCC_I2C1_CLK_ENABLE();
    }
}

void HAL_SPI_MspInit(SPI_HandleTypeDef *hspi)
{
    GPIO_InitTypeDef g = {0};
    if (hspi->Instance == SPI1) {
        __HAL_RCC_SPI1_CLK_ENABLE();
        __HAL_RCC_GPIOA_CLK_ENABLE();
        g.Pin = GPIO_PIN_5 | GPIO_PIN_7;           /* PA5 SCK，PA7 MOSI */
        g.Mode = GPIO_MODE_AF_PP;
        g.Speed = GPIO_SPEED_FREQ_HIGH;
        HAL_GPIO_Init(GPIOA, &g);
        g.Pin = GPIO_PIN_6;                        /* PA6 MISO */
        g.Mode = GPIO_MODE_INPUT;
        g.Pull = GPIO_NOPULL;
        HAL_GPIO_Init(GPIOA, &g);
    }
}

void HAL_UART_MspInit(UART_HandleTypeDef *huart)
{
    GPIO_InitTypeDef g = {0};
    if (huart->Instance == USART1) {
        __HAL_RCC_USART1_CLK_ENABLE();
        __HAL_RCC_GPIOA_CLK_ENABLE();
        g.Pin = GPIO_PIN_9;                        /* PA9 TX */
        g.Mode = GPIO_MODE_AF_PP;
        g.Speed = GPIO_SPEED_FREQ_HIGH;
        HAL_GPIO_Init(GPIOA, &g);
        g.Pin = GPIO_PIN_10;                       /* PA10 RX */
        g.Mode = GPIO_MODE_INPUT;
        g.Pull = GPIO_NOPULL;
        HAL_GPIO_Init(GPIOA, &g);
    }
}
