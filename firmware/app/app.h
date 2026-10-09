/*
 * app.h —— 应用层入口（在 CubeMX 生成的 main.c 中调用）
 *
 * 在 main.c 中加入：
 *   / * USER CODE BEGIN Includes * /   #include "app.h"
 *   / * USER CODE BEGIN 2 * /          app_init();
 *   / * USER CODE BEGIN 3 * /          app_loop();     （放在 while(1) 循环体内）
 *
 * 以下参数都可以在 CubeIDE 的编译选项里用 -D 覆盖，一般不需要改。
 */
#ifndef APP_H
#define APP_H

#include <stdint.h>

#ifndef APP_AXIS
#define APP_AXIS 2          /* 在线诊断 / 演示使用的加速度轴：0=X 1=Y 2=Z，须与 Python config.FAN_AXIS 一致 */
#endif

#ifndef APP_POLL_MS
#define APP_POLL_MS 20      /* 空闲时每隔多少毫秒检查一次按键 */
#endif

#ifndef APP_DISP_MS
#define APP_DISP_MS 200     /* 与电脑通信时 OLED 最快刷新间隔（毫秒） */
#endif

#ifndef APP_DEMO_N
#define APP_DEMO_N 32       /* 演示模式下每层校准使用的窗口数 */
#endif

#ifndef APP_DEMO_LAYERS
#define APP_DEMO_LAYERS 1   /* 演示模式校准的层数：1 = 论文方法 M3L1（只校准第一层）；改为 7 即全部层（M3） */
#endif

#ifndef APP_KEY_ACTIVE
#define APP_KEY_ACTIVE GPIO_PIN_RESET   /* 按键按下时 PB0 的电平；按键模块按下输出高电平时改为 GPIO_PIN_SET */
#endif

void app_init(void);
void app_loop(void);                 /* 等待并处理一条电脑命令；空闲时检查按键 */
void app_delay_us(uint32_t us);      /* 基于 DWT 周期计数器的微秒延时 */

#endif
