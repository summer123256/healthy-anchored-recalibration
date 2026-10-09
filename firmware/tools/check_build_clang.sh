#!/bin/sh
# 用 clang + ld.lld 把完整固件（参考 main.c + HAL + 启动文件 + 链接脚本 + app + 模型）交叉编译并链接到 Cortex-M3，
# 检查：能否编译通过、能否链接、Flash / RAM 是否放得下。
# 用法（Linux）：sh check_build_clang.sh [模型目录，默认 ../model]
# 说明：这是我在没有 arm-none-eabi-gcc 的环境下做的验证；你用 CubeIDE 编译时不需要它。
set -e
cd "$(dirname "$0")"
FW=..
MODEL=${1:-$FW/model}
P=$FW/project_make
OUT=build_clang
rm -rf $OUT && mkdir -p $OUT
CFLAGS="--target=thumbv7m-none-eabi -mcpu=cortex-m3 -mthumb -mfloat-abi=soft -O2 -ffreestanding \
 -ffunction-sections -fdata-sections -Wall -Wno-unused-parameter -DUSE_HAL_DRIVER -DSTM32F103xB \
 -Iclang_shim -I$P/Core/Inc -I$FW/app -I$MODEL -I$P/Drivers/STM32F1xx_HAL_Driver/Inc \
 -I$P/Drivers/STM32F1xx_HAL_Driver/Inc/Legacy -I$P/Drivers/CMSIS/Device/ST/STM32F1xx/Include \
 -I$P/Drivers/CMSIS/Include"
SRCS="$P/Core/Src/main.c $P/Core/Src/stm32f1xx_it.c $P/Core/Src/stm32f1xx_hal_msp.c \
 $P/Core/Src/system_stm32f1xx.c $FW/app/app.c $FW/app/adxl345.c $FW/app/ssd1306.c $FW/app/tinycnn.c \
 $MODEL/model_data.c clang_shim/rt_shim.c $(ls $P/Drivers/STM32F1xx_HAL_Driver/Src/*.c)"
for s in $SRCS; do
  clang $CFLAGS -c "$s" -o "$OUT/$(basename "$s" .c).o"
done
grep -v "^ *\.fpu" $P/startup_stm32f103xb.s > $OUT/startup.s
clang --target=thumbv7m-none-eabi -mcpu=cortex-m3 -mthumb -c $OUT/startup.s -o $OUT/startup.o
ld.lld -T $P/STM32F103C8Tx_FLASH.ld --gc-sections -Map=$OUT/fw.map $OUT/*.o -o $OUT/fw.elf
llvm-objcopy -O ihex $OUT/fw.elf $OUT/fw.hex
llvm-size -A $OUT/fw.elf | grep -E "isr_vector|\.text|\.rodata|\.data|\.bss|user_heap|Total"
llvm-size $OUT/fw.elf
