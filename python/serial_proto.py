"""
电脑 <-> STM32 串口通信协议（与 firmware/app/app.c 一致）
=====================================================
帧格式：0xA5 0x5A | 命令(1B) | 长度(2B, 小端) | 数据 | 校验(1B)
校验 = (命令 + 长度低字节 + 长度高字节 + 所有数据字节) & 0xFF

电脑 -> 单片机：
  0x01 INFER         数据 = 1024 个 int16（小端，2048 字节），单片机做预处理+推理
  0x02 STREAM_START  开始连续发送 ADXL345 三轴原始数据（采集风扇数据集用）
  0x03 STOP          停止连续发送 / 停止在线诊断
  0x04 ONLINE_START  开始在线诊断（单片机自己采集 1024 点 -> 推理 -> 发送结果，循环）
  0x05 SELFTEST      运行片上自检测试向量
  0x06 INFO          查询模型信息
  0x07 CALIBRATE     单片机自主片上校准：N(u8) anchor(u8: 0 健康锚点 / 1 全部类别) scale(u8) 层数(u8, 0=全部)
  0x08 CALIB_FEED    硬件在环校准：层号 k(u8) + 1024 个 int16，累加到第 k 个可校准层
  0x09 CALIB_APPLY   k(u8) anchor(u8) scale(u8)：用已累加的统计量更新第 k 层并清零累加器
  0x0A RESET_PARAMS  恢复出厂参数（清除校准）
  0x0B DUMP          k(u8)：读取第 k 层当前的偏置、乘数、移位
单片机 -> 电脑：
  0x81 INFER_RESULT  pred(u8) cyc_pre(u32) cyc_inf(u32) logits(int32 × 类别数)
  0x82 STREAM_DATA   seq(u16) n(u8) overrun(u8) 然后 n 组 (x,y,z) int16
  0x84 ONLINE_RESULT 与 0x81 相同
  0x85 SELFTEST_RES  pass(u8) total(u8) cyc_inf(u32)
  0x86 INFO          ASCII 字符串
  0x87 CALIBRATE     status(i8) 使用窗口数(u16) 总耗时毫秒(u32) 计算周期数(u32)
  0x88 CALIB_FEED    status(i8) 已累加窗口数(u16) 本窗口预处理+前向+累加的周期数(u32)
  0x89 CALIB_APPLY   status(i8) 周期数(u32)
  0x8A RESET_PARAMS  （无数据）
  0x8B DUMP          通道数 n(u8) + b[n](int32) + mult[n](int32) + shift[n](int8)
  0xEE ERROR         错误码(u8)
"""
import struct

SYNC = b"\xA5\x5A"
CMD_INFER, CMD_STREAM, CMD_STOP, CMD_ONLINE, CMD_SELFTEST, CMD_INFO = 0x01, 0x02, 0x03, 0x04, 0x05, 0x06
CMD_CALIBRATE, CMD_CALIB_FEED, CMD_CALIB_APPLY, CMD_RESET_PARAMS, CMD_DUMP = 0x07, 0x08, 0x09, 0x0A, 0x0B
RSP_INFER, RSP_STREAM, RSP_ONLINE, RSP_SELFTEST, RSP_INFO, RSP_ERROR = 0x81, 0x82, 0x84, 0x85, 0x86, 0xEE
RSP_CALIBRATE, RSP_CALIB_FEED, RSP_CALIB_APPLY, RSP_RESET_PARAMS, RSP_DUMP = 0x87, 0x88, 0x89, 0x8A, 0x8B
ERRORS = {1: "校验错误", 2: "长度错误", 3: "未知命令", 4: "加速度计未连接", 5: "校准参数错误"}


def checksum(cmd, payload):
    n = len(payload)
    return (cmd + (n & 0xFF) + (n >> 8) + sum(payload)) & 0xFF


def make_frame(cmd, payload=b""):
    return SYNC + struct.pack("<BH", cmd, len(payload)) + payload + bytes([checksum(cmd, payload)])


def read_exact(ser, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = ser.read(n - len(buf))
        if not chunk:
            raise TimeoutError("串口读取超时")
        buf += chunk
    return bytes(buf)


def read_frame(ser):
    """阻塞读取一帧，返回 (cmd, payload)。自动跳过不同步的字节。"""
    state = 0
    while True:
        b = read_exact(ser, 1)[0]
        if state == 0:
            state = 1 if b == 0xA5 else 0
        elif state == 1:
            if b == 0x5A:
                break
            state = 1 if b == 0xA5 else 0
    cmd, n = struct.unpack("<BH", read_exact(ser, 3))
    payload = read_exact(ser, n)
    cs = read_exact(ser, 1)[0]
    if cs != checksum(cmd, payload):
        raise ValueError(f"校验错误（命令 0x{cmd:02X}）")
    return cmd, payload


def parse_result(payload, n_classes):
    pred, cyc_pre, cyc_inf = struct.unpack_from("<BII", payload, 0)
    logits = struct.unpack_from(f"<{n_classes}i", payload, 9)
    return pred, cyc_pre, cyc_inf, list(logits)


def expect(ser, rsp):
    """读取一帧并确认是期望的响应；收到错误帧时抛出异常。"""
    cmd, pl = read_frame(ser)
    if cmd == RSP_ERROR:
        raise RuntimeError(f"单片机返回错误：{ERRORS.get(pl[0], pl[0])}")
    if cmd != rsp:
        raise RuntimeError(f"期望响应 0x{rsp:02X}，实际收到 0x{cmd:02X}")
    return pl


def parse_dump(pl):
    n = pl[0]
    b = struct.unpack_from(f"<{n}i", pl, 1)
    m = struct.unpack_from(f"<{n}i", pl, 1 + 4 * n)
    s = struct.unpack_from(f"<{n}b", pl, 1 + 8 * n)
    return list(b), list(m), list(s)


class VirtualSerial:
    """“虚拟单片机”：在电脑上运行用 gcc 编译的固件（firmware/host_test/virtual_mcu.c），
    通过标准输入输出通信。没有硬件时可用它测试全部串口脚本（仅 Linux / macOS / WSL）。"""

    def __init__(self, exe, timeout=3.0):
        import subprocess
        self.timeout = timeout
        self.p = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

    def write(self, data):
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def read(self, n):
        import os
        import select
        buf = b""
        fd = self.p.stdout.fileno()
        while len(buf) < n:
            r, _, _ = select.select([fd], [], [], self.timeout)
            if not r:
                break
            chunk = os.read(fd, n - len(buf))
            if not chunk:
                break
            buf += chunk
        return buf

    def reset_input_buffer(self):
        import os
        import select
        fd = self.p.stdout.fileno()
        while select.select([fd], [], [], 0.05)[0]:
            if not os.read(fd, 4096):
                break

    def close(self):
        try:
            self.p.stdin.close()
        except OSError:
            pass
        self.p.terminate()
        self.p.wait(timeout=5)


def serial_open(port, baud, timeout=3.0):
    """打开串口。port 写成 "virtual:<模型导出目录>" 时使用虚拟单片机（需先运行 build_virtual_mcu.py）。"""
    if port.startswith("virtual:"):
        from pathlib import Path
        exe = Path(port[len("virtual:"):]) / "virtual_mcu"
        if not exe.exists():
            raise FileNotFoundError(f"找不到 {exe}，请先运行 python build_virtual_mcu.py --model-dir {exe.parent}")
        return VirtualSerial(exe, timeout)
    import serial
    return serial.Serial(port, baud, timeout=timeout)
