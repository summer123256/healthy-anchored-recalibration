/*
 * stm32f1xx_it.c —— 中断服务函数（与 CubeMX 生成的内容等价；本项目只用 SysTick）
 */
#include "main.h"

void NMI_Handler(void)        { while (1) { } }
void HardFault_Handler(void)  { while (1) { } }
void MemManage_Handler(void)  { while (1) { } }
void BusFault_Handler(void)   { while (1) { } }
void UsageFault_Handler(void) { while (1) { } }
void SVC_Handler(void)        { }
void DebugMon_Handler(void)   { }
void PendSV_Handler(void)     { }

void SysTick_Handler(void)
{
    HAL_IncTick();             /* HAL_Delay / HAL_GetTick / 各种超时都依赖它 */
}
