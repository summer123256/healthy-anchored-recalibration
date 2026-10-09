/* 电脑端模拟用的最小 HAL（仅用于在 PC 上编译/仿真 app.c，不用于单片机） */
#ifndef MOCK_MAIN_H
#define MOCK_MAIN_H
#include <stdint.h>
typedef enum { HAL_OK = 0, HAL_ERROR, HAL_BUSY, HAL_TIMEOUT } HAL_StatusTypeDef;
typedef enum { GPIO_PIN_RESET = 0, GPIO_PIN_SET } GPIO_PinState;
typedef struct { uint32_t dummy; } GPIO_TypeDef;
typedef struct { uint32_t dummy; } SPI_HandleTypeDef;
typedef struct { volatile uint32_t SR, DR; } USART_TypeDef;
typedef struct { USART_TypeDef *Instance; } UART_HandleTypeDef;
typedef struct { volatile uint32_t DEMCR; } CoreDebug_Type;
typedef struct { volatile uint32_t CTRL, CYCCNT; } DWT_Type;
extern GPIO_TypeDef mock_gpioa; extern CoreDebug_Type mock_cd; extern DWT_Type mock_dwt;
extern uint32_t SystemCoreClock;
#define GPIOA (&mock_gpioa)
#define GPIO_PIN_4 0x10
#define HAL_MAX_DELAY 0xFFFFFFFFU
#define CoreDebug (&mock_cd)
DWT_Type *mock_dwt_tick(void);   /* 每次访问 DWT 时计数器前进，避免延时死循环 */
#define DWT (mock_dwt_tick())
#define CoreDebug_DEMCR_TRCENA_Msk (1u << 24)
#define DWT_CTRL_CYCCNTENA_Msk 1u
#define UART_FLAG_RXNE 0x20
#define UART_FLAG_ORE 0x08
int mock_uart_flag(UART_HandleTypeDef *h, uint32_t f);
#define __HAL_UART_GET_FLAG(h, f) mock_uart_flag((h), (f))
void HAL_GPIO_WritePin(GPIO_TypeDef *p, uint16_t pin, GPIO_PinState s);
HAL_StatusTypeDef HAL_SPI_Transmit(SPI_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t);
HAL_StatusTypeDef HAL_SPI_Receive(SPI_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t);
HAL_StatusTypeDef HAL_UART_Transmit(UART_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t);
HAL_StatusTypeDef HAL_UART_Receive(UART_HandleTypeDef *h, uint8_t *d, uint16_t n, uint32_t t);
void HAL_Delay(uint32_t ms);
uint32_t HAL_GetTick(void);
#endif
