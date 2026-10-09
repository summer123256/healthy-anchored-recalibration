# ============================================================================
# CLion / CMake 工程：STM32F103C8T6 故障诊断固件（与 Makefile 等价）
#
# CLion：File → Open → 选择本文件夹（project_make），按提示加载 CMake 工程。
#   工具链：Settings → Build, Execution, Deployment → Toolchains 里新建一个，
#           C Compiler / C++ Compiler 指向 arm-none-eabi-gcc / arm-none-eabi-g++；
#   或在 CMake 选项里加：-DCMAKE_TOOLCHAIN_FILE=arm-gcc-toolchain.cmake
#
# 换成训练后导出的模型：CMake 选项加 -DMODEL_DIR=<导出目录的绝对路径>，例如
#   -DMODEL_DIR=D:/stm32_fault_tinyml/python/export/pu_tiny
# 不加时使用 ../model 里的占位模型（随机权重，只用来测试烧录和串口）。
#
# 输出：fault_tinyml.elf / .hex / .bin（在 CLion 的 cmake-build-* 目录里）
# 烧录：CLion 的 “OpenOCD Download & Run”，Board config file 选本文件夹的 openocd_stm32f103.cfg
# ============================================================================
cmake_minimum_required(VERSION 3.16)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

project(fault_tinyml C ASM)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../model" CACHE PATH "模型目录（含 model_data.c / model_data.h）")
set(APP_DIR   "${CMAKE_CURRENT_SOURCE_DIR}/../app")
set(HAL       "${CMAKE_CURRENT_SOURCE_DIR}/Drivers/STM32F1xx_HAL_Driver/Src")
message(STATUS "使用的模型目录：${MODEL_DIR}")

set(MCU_FLAGS -mcpu=cortex-m3 -mthumb)

add_executable(${PROJECT_NAME}.elf
  Core/Src/main.c Core/Src/stm32f1xx_it.c Core/Src/stm32f1xx_hal_msp.c Core/Src/system_stm32f1xx.c
  ${APP_DIR}/app.c ${APP_DIR}/adxl345.c ${APP_DIR}/ssd1306.c ${APP_DIR}/tinycnn.c
  ${MODEL_DIR}/model_data.c
  ${HAL}/stm32f1xx_hal.c ${HAL}/stm32f1xx_hal_cortex.c ${HAL}/stm32f1xx_hal_dma.c
  ${HAL}/stm32f1xx_hal_exti.c ${HAL}/stm32f1xx_hal_flash.c ${HAL}/stm32f1xx_hal_flash_ex.c
  ${HAL}/stm32f1xx_hal_gpio.c ${HAL}/stm32f1xx_hal_gpio_ex.c ${HAL}/stm32f1xx_hal_i2c.c
  ${HAL}/stm32f1xx_hal_pwr.c ${HAL}/stm32f1xx_hal_rcc.c ${HAL}/stm32f1xx_hal_rcc_ex.c
  ${HAL}/stm32f1xx_hal_spi.c ${HAL}/stm32f1xx_hal_uart.c
  startup_stm32f103xb.s
  ${EXTRA_SOURCES}
)

target_compile_definitions(${PROJECT_NAME}.elf PRIVATE USE_HAL_DRIVER STM32F103xB)
target_include_directories(${PROJECT_NAME}.elf PRIVATE
  ${EXTRA_INCLUDES}
  Core/Inc ${APP_DIR} ${MODEL_DIR}
  Drivers/STM32F1xx_HAL_Driver/Inc Drivers/STM32F1xx_HAL_Driver/Inc/Legacy
  Drivers/CMSIS/Device/ST/STM32F1xx/Include Drivers/CMSIS/Include)

# 与 Makefile 相同：-O2（论文 4.4 节），不论 Debug / Release
target_compile_options(${PROJECT_NAME}.elf PRIVATE
  ${MCU_FLAGS} -O2 -g -Wall -ffunction-sections -fdata-sections
  $<$<COMPILE_LANGUAGE:ASM>:-x assembler-with-cpp>)

set(LDSCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/STM32F103C8Tx_FLASH.ld)
if(CMAKE_C_COMPILER_ID STREQUAL "GNU")
  target_link_options(${PROJECT_NAME}.elf PRIVATE ${MCU_FLAGS}
    -specs=nano.specs -specs=nosys.specs -T${LDSCRIPT}
    -Wl,-Map=${PROJECT_NAME}.map,--cref -Wl,--gc-sections -Wl,--print-memory-usage)
  target_link_libraries(${PROJECT_NAME}.elf PRIVATE c m)
else()
  target_link_options(${PROJECT_NAME}.elf PRIVATE ${MCU_FLAGS} -T${LDSCRIPT} -Wl,--gc-sections
    -Wl,-Map=${PROJECT_NAME}.map ${EXTRA_LINK_OPTIONS})
endif()

# 生成 .hex / .bin，并打印 Flash / RAM 用量（论文表 11）
set(OBJCOPY_EXE ${CMAKE_OBJCOPY})
set(SIZE_EXE ${CMAKE_SIZE})
if(NOT SIZE_EXE)
  string(REGEX REPLACE "gcc(\\.exe)?$" "size\\1" SIZE_EXE "${CMAKE_C_COMPILER}")
endif()
add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
  COMMAND ${OBJCOPY_EXE} -O ihex $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.hex
  COMMAND ${OBJCOPY_EXE} -O binary -S $<TARGET_FILE:${PROJECT_NAME}.elf> ${PROJECT_NAME}.bin
  COMMAND ${SIZE_EXE} $<TARGET_FILE:${PROJECT_NAME}.elf>
  COMMENT "生成 ${PROJECT_NAME}.hex / .bin；上面 text+data = Flash 用量，data+bss = 静态 RAM 用量")
