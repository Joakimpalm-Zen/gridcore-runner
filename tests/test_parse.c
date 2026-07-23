// RNR-021: the one strict numeric parser shared by CLI flags and env overrides
// must reject what the old atoi/atof/strtoull silently turned into 0 or a
// wrapped value — trailing garbage, empty, overflow, sign errors, and (for
// doubles) NaN/inf. A bad setting must not disable a deadline or overflow a
// byte budget.
#include "compat.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static void test_i64(void) {
    long long v = 123;
    assert(parse_i64("42", 0, 100, &v) && v == 42);
    assert(parse_i64("-5", -10, 10, &v) && v == -5);
    v = 777;                                      // sentinel for the failure cases
    assert(!parse_i64("", 0, 100, &v));           // empty
    assert(!parse_i64("42x", 0, 100, &v));        // trailing garbage
    assert(!parse_i64("101", 0, 100, &v));        // above range
    assert(!parse_i64("-1", 0, 100, &v));         // below range
    assert(!parse_i64("99999999999999999999", 0, INT64_MAX, &v)); // overflow
    assert(v == 777);                             // *out untouched on failure
    printf("ok: parse_i64\n");
}

static void test_u64(void) {
    uint64_t v = 7;
    assert(parse_u64("512", 0, 1u << 20, &v) && v == 512);
    v = 777;
    assert(!parse_u64("-1", 0, UINT64_MAX, &v));  // strtoull would wrap this
    assert(!parse_u64("3.5", 0, 100, &v));        // trailing garbage
    assert(!parse_u64("", 0, 100, &v));
    assert(!parse_u64("2000000", 0, 1u << 20, &v)); // above range
    assert(!parse_u64("99999999999999999999999", 0, UINT64_MAX, &v)); // overflow
    assert(v == 777);
    printf("ok: parse_u64\n");
}

static void test_f64(void) {
    double v = 1.0;
    assert(parse_f64("3.5", 0.0, 10.0, &v) && v == 3.5);
    assert(parse_f64("0", 0.0, 10.0, &v) && v == 0.0);   // an explicit 0 is valid
    v = 777.0;
    assert(!parse_f64("nan", 0.0, 10.0, &v));            // NaN rejected
    assert(!parse_f64("inf", 0.0, 1e18, &v));            // inf rejected
    assert(!parse_f64("abc", 0.0, 10.0, &v));
    assert(!parse_f64("1.0e", 0.0, 10.0, &v));           // trailing garbage
    assert(!parse_f64("", 0.0, 10.0, &v));
    assert(!parse_f64("11", 0.0, 10.0, &v));             // above range
    assert(v == 777.0);
    printf("ok: parse_f64\n");
}

int main(void) {
    test_i64();
    test_u64();
    test_f64();
    printf("all parse tests passed\n");
    return 0;
}
