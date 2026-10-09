/*
 * rt_shim.c —— 仅用于 clang + ld.lld 验证链接（本环境没有 ARM 版 libgcc / newlib）。
 * 用 CubeIDE 或 arm-none-eabi-gcc 编译时不需要这个文件：这些函数由 newlib 和 libgcc 提供。
 */
#include <stddef.h>
#include <stdint.h>

void *memcpy(void *d, const void *s, size_t n) { uint8_t *p = d; const uint8_t *q = s; while (n--) *p++ = *q++; return d; }
void *memset(void *d, int c, size_t n) { uint8_t *p = d; while (n--) *p++ = (uint8_t)c; return d; }
void *memmove(void *d, const void *s, size_t n)
{
    uint8_t *p = d; const uint8_t *q = s;
    if (p < q) while (n--) *p++ = *q++; else { p += n; q += n; while (n--) *--p = *--q; }
    return d;
}
int memcmp(const void *a, const void *b, size_t n)
{
    const uint8_t *p = a, *q = b;
    for (; n; n--, p++, q++) if (*p != *q) return *p - *q;
    return 0;
}
size_t strlen(const char *s) { size_t n = 0; while (s[n]) n++; return n; }
void __aeabi_memcpy(void *d, const void *s, size_t n) { memcpy(d, s, n); }
void __aeabi_memcpy4(void *d, const void *s, size_t n) { memcpy(d, s, n); }
void __aeabi_memcpy8(void *d, const void *s, size_t n) { memcpy(d, s, n); }
void __aeabi_memmove(void *d, const void *s, size_t n) { memmove(d, s, n); }
void __aeabi_memmove4(void *d, const void *s, size_t n) { memmove(d, s, n); }
void __aeabi_memset(void *d, size_t n, int c) { memset(d, c, n); }
void __aeabi_memset4(void *d, size_t n, int c) { memset(d, c, n); }
void __aeabi_memclr(void *d, size_t n) { memset(d, 0, n); }
void __aeabi_memclr4(void *d, size_t n) { memset(d, 0, n); }
void __aeabi_memclr8(void *d, size_t n) { memset(d, 0, n); }
void __libc_init_array(void) { }

/* 64 位无符号除法（移位相减），供 __aeabi_uldivmod / __aeabi_ldivmod 使用 */
uint64_t shim_udivmod64(uint64_t n, uint64_t d, uint64_t *rem)
{
    uint64_t q = 0, r = 0;
    for (int i = 63; i >= 0; i--) {
        r = (r << 1) | ((n >> i) & 1u);
        if (r >= d) { r -= d; q |= (uint64_t)1 << i; }
    }
    *rem = r;
    return q;
}
int64_t shim_divmod64(int64_t n, int64_t d, int64_t *rem)
{
    uint64_t un = n < 0 ? (uint64_t)0 - (uint64_t)n : (uint64_t)n;
    uint64_t ud = d < 0 ? (uint64_t)0 - (uint64_t)d : (uint64_t)d;
    uint64_t ur;
    uint64_t uq = shim_udivmod64(un, ud, &ur);
    *rem = n < 0 ? -(int64_t)ur : (int64_t)ur;
    return ((n < 0) != (d < 0)) ? -(int64_t)uq : (int64_t)uq;
}
/* AEABI 约定：参数 r0:r1 / r2:r3，返回商 r0:r1、余数 r2:r3 */
__attribute__((naked)) void __aeabi_uldivmod(void)
{
    __asm volatile("push {r4, lr}\n sub sp, sp, #16\n add r4, sp, #8\n str r4, [sp]\n"
                   "bl shim_udivmod64\n ldrd r2, r3, [sp, #8]\n add sp, sp, #16\n pop {r4, pc}\n");
}
__attribute__((naked)) void __aeabi_ldivmod(void)
{
    __asm volatile("push {r4, lr}\n sub sp, sp, #16\n add r4, sp, #8\n str r4, [sp]\n"
                   "bl shim_divmod64\n ldrd r2, r3, [sp, #8]\n add sp, sp, #16\n pop {r4, pc}\n");
}
