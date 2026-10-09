# 电脑端验证：用 gcc 编译单片机代码并与 Python 结果逐位比对
#   make MODEL_DIR=../../python/export/tiny
#   ./test_host  ../../python/export/tiny/vectors.bin          # 推理逐位比对
#   ./test_calib <calib_vectors.bin>                           # 片上重校准逐位比对（selftest.py 生成）
# 一般直接运行 python/selftest.py 即可，它会自动完成全部编译与比对。
MODEL_DIR ?= ../../python/export/tiny
CC ?= gcc
CFLAGS ?= -O2 -std=c99 -Wall -Wextra

all: test_host test_calib

test_host: test_host.c ../app/tinycnn.c $(MODEL_DIR)/model_data.c
	$(CC) $(CFLAGS) -I$(MODEL_DIR) -I../app -o $@ $^

test_calib: test_calib.c ../app/tinycnn.c $(MODEL_DIR)/model_data.c
	$(CC) $(CFLAGS) -I$(MODEL_DIR) -I../app -o $@ $^

clean:
	rm -f test_host test_calib

.PHONY: all clean
