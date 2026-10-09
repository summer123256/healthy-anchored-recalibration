# ============================================================================
# 用 Makefile 直接编译 STM32F103C8T6 固件（不经过 CubeIDE 的另一种方式）
#
# 需要：arm-none-eabi-gcc（ARM GNU Toolchain；STM32CubeIDE 安装目录里也自带一份）和 make
#
#   make                                   # 使用 ../model 里的占位模型（随机权重，只测流程）
#   make MODEL_DIR=../../python/export/tiny   # 使用你训练后导出的模型
#   make clean
#
# 输出：build/fault_tinyml.elf / .hex / .bin
# 烧录：用 STM32CubeProgrammer（免费）打开 .hex，经 ST-Link 下载；或 st-flash write build/fault_tinyml.bin 0x08000000
# ============================================================================
TARGET    = fault_tinyml
BUILD     = build
MODEL_DIR ?= ../model
APP_DIR   = ../app
PREFIX   ?= arm-none-eabi-
CC        = $(PREFIX)gcc
AS        = $(PREFIX)gcc -x assembler-with-cpp
CP        = $(PREFIX)objcopy
SZ        = $(PREFIX)size

HAL = Drivers/STM32F1xx_HAL_Driver/Src
C_SOURCES = \
  Core/Src/main.c Core/Src/stm32f1xx_it.c Core/Src/stm32f1xx_hal_msp.c Core/Src/system_stm32f1xx.c \
  $(APP_DIR)/app.c $(APP_DIR)/adxl345.c $(APP_DIR)/ssd1306.c $(APP_DIR)/tinycnn.c \
  $(MODEL_DIR)/model_data.c \
  $(HAL)/stm32f1xx_hal.c $(HAL)/stm32f1xx_hal_cortex.c $(HAL)/stm32f1xx_hal_dma.c \
  $(HAL)/stm32f1xx_hal_exti.c $(HAL)/stm32f1xx_hal_flash.c $(HAL)/stm32f1xx_hal_flash_ex.c \
  $(HAL)/stm32f1xx_hal_gpio.c $(HAL)/stm32f1xx_hal_gpio_ex.c $(HAL)/stm32f1xx_hal_i2c.c \
  $(HAL)/stm32f1xx_hal_pwr.c $(HAL)/stm32f1xx_hal_rcc.c $(HAL)/stm32f1xx_hal_rcc_ex.c \
  $(HAL)/stm32f1xx_hal_spi.c $(HAL)/stm32f1xx_hal_uart.c
ASM_SOURCES = startup_stm32f103xb.s

MCU  = -mcpu=cortex-m3 -mthumb
DEFS = -DUSE_HAL_DRIVER -DSTM32F103xB
INCS = -ICore/Inc -I$(APP_DIR) -I$(MODEL_DIR) \
       -IDrivers/STM32F1xx_HAL_Driver/Inc -IDrivers/STM32F1xx_HAL_Driver/Inc/Legacy \
       -IDrivers/CMSIS/Device/ST/STM32F1xx/Include -IDrivers/CMSIS/Include
CFLAGS  = $(MCU) $(DEFS) $(INCS) -O2 -std=gnu11 -Wall -ffunction-sections -fdata-sections -g
LDFLAGS = $(MCU) -specs=nano.specs -specs=nosys.specs -TSTM32F103C8Tx_FLASH.ld \
          -Wl,-Map=$(BUILD)/$(TARGET).map,--cref -Wl,--gc-sections -Wl,--print-memory-usage -lc -lm

OBJECTS  = $(addprefix $(BUILD)/,$(notdir $(C_SOURCES:.c=.o)))
OBJECTS += $(addprefix $(BUILD)/,$(notdir $(ASM_SOURCES:.s=.o)))
vpath %.c $(sort $(dir $(C_SOURCES)))
vpath %.s $(sort $(dir $(ASM_SOURCES)))

all: $(BUILD)/$(TARGET).elf $(BUILD)/$(TARGET).hex $(BUILD)/$(TARGET).bin

$(BUILD)/%.o: %.c Makefile | $(BUILD)
	$(CC) -c $(CFLAGS) $< -o $@
$(BUILD)/%.o: %.s Makefile | $(BUILD)
	$(AS) -c $(MCU) $< -o $@
$(BUILD)/$(TARGET).elf: $(OBJECTS) Makefile
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	$(SZ) $@
$(BUILD)/%.hex: $(BUILD)/%.elf
	$(CP) -O ihex $< $@
$(BUILD)/%.bin: $(BUILD)/%.elf
	$(CP) -O binary -S $< $@
$(BUILD):
	mkdir -p $@

clean:
	rm -rf $(BUILD)

.PHONY: all clean
