/* 占位模型（随机权重，含源域统计量）：只用于先测试烧录、串口、推理与校准耗时。训练后用 quantize_export.py 导出的文件覆盖。 */
#ifndef MODEL_DATA_H
#define MODEL_DATA_H
#include "tinycnn.h"

#define TC_MODEL_NAME    "tiny"
#define TC_INPUT_LEN     1024
#define TC_NUM_CLASSES   10
#define TC_NUM_LAYERS    12
#define TC_MAX_ACT       2048   /* 单个激活缓冲区字节数（共用两个，乒乓） */
#define TC_NUM_CALIB     7   /* 可校准卷积层个数 */
#define TC_CALIB_CH      224   /* 可校准通道总数 */
#define TC_MAX_CALIB_CH  64   /* 单层最大通道数 */
#define TC_HAS_STATS     1   /* 是否包含源域统计量 */

extern const tc_layer_t tc_layers[TC_NUM_LAYERS];
extern const uint8_t tc_calib_layers[7];
extern const char *const tc_class_names[TC_NUM_CLASSES];

#endif
