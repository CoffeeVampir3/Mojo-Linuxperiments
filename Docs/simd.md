# Mojo `SIMD` struct

```mojo
struct SIMD[dtype: DType, size: Int]
```

Represents a vector type that leverages hardware acceleration to process multiple data elements with a single operation.

SIMD (Single Instruction, Multiple Data) is a fundamental parallel computing paradigm where a single CPU instruction operates on multiple data elements at once. Modern CPUs can perform 4, 8, 16, or even 32 operations in parallel using SIMD, delivering substantial performance improvements over scalar operations. Instead of processing one value at a time, SIMD processes entire vectors of values with each instruction.

For example, when adding two vectors of four values, a scalar operation adds each value in the vector one by one, while a SIMD operation adds all four values at once using vector registers:

```
Scalar operation:                SIMD operation:
┌─────────────────────────┐      ┌───────────────────────────┐
│ 4 instructions          │      │ 1 instruction             │
│ 4 clock cycles          │      │ 1 clock cycle             │
│                         │      │                           │
│ ADD  a[0], b[0] → c[0]  │      │ Vector register A         │
│ ADD  a[1], b[1] → c[1]  │      │ ┌─────┬─────┬─────┬─────┐ │
│ ADD  a[2], b[2] → c[2]  │      │ │a[0] │a[1] │a[2] │a[3] │ │
│ ADD  a[3], b[3] → c[3]  │      │ └─────┴─────┴─────┴─────┘ │
└─────────────────────────┘      │           +               │
                                 │ Vector register B         │
                                 │ ┌─────┬─────┬─────┬─────┐ │
                                 │ │b[0] │b[1] │b[2] │b[3] │ │
                                 │ └─────┴─────┴─────┴─────┘ │
                                 │           ↓               │
                                 │        SIMD_ADD           │
                                 │           ↓               │
                                 │ Vector register C         │
                                 │ ┌─────┬─────┬─────┬─────┐ │
                                 │ │c[0] │c[1] │c[2] │c[3] │ │
                                 │ └─────┴─────┴─────┴─────┘ │
                                 └───────────────────────────┘
```

The SIMD type maps directly to hardware vector registers and instructions. Mojo automatically generates optimal SIMD code that leverages CPU-specific instruction sets (such as AVX and NEON) without requiring manual intrinsics or assembly programming.

This type is the foundation of high-performance CPU computing in Mojo, enabling you to write code that automatically leverages modern CPU vector capabilities while maintaining code clarity and portability.

> **Caution:** If you declare a SIMD vector size larger than the vector registers of the target hardware, the compiler will break up the SIMD into multiple vector registers for compatibility. However, you should avoid using a vector that's more than 2x the hardware's vector register size because the resulting code will perform poorly.

## Key properties

- **Hardware-mapped:** Directly maps to CPU vector registers
- **Type-safe:** Data types and vector sizes are checked at compile time
- **Zero-cost:** No runtime overhead compared to hand-optimized intrinsics
- **Portable:** Same code works across different CPU architectures (x86, ARM, etc.)
- **Composable:** Seamlessly integrates with Mojo's parallelization features

## Key APIs

**Construction:**
- Broadcast single value to all elements: `SIMD[dtype, size](value)`
- Initialize with specific values: `SIMD[dtype, size](v1, v2, ...)`
- Zero-initialized vector: `SIMD[dtype, size]()`

**Element operations:**
- Arithmetic: `+`, `-`, `*`, `/`, `%`, `//`
- Comparison: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Math functions: `sqrt()`, `sin()`, `cos()`, `fma()`, etc.
- Bit operations: `&`, `|`, `^`, `~`, `<<`, `>>`

**Vector operations:**
- Horizontal reductions: `reduce_add()`, `reduce_mul()`, `reduce_min()`, `reduce_max()`
- Element-wise conditional selection: `select(condition, true_case, false_case)`
- Vector manipulation: `shuffle()`, `slice()`, `join()`, `split()`
- Type conversion: `cast[target_dtype]()`

## Examples

**Vectorized math operations:**

```mojo
# Process 8 floating-point numbers simultaneously
var a = SIMD[DType.float32, 8](1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
var b = SIMD[DType.float32, 8](2.0)  # Broadcast 2.0 to all elements
var result = a * b + 1.0
print(result)  # => [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]
```

**Conditional operations with masking:**

```mojo
# Double the positive values and negate the negative values
var values = SIMD[DType.int32, 4](1, -2, 3, -4)
var is_positive = values.gt(0)  # greater-than: gets SIMD of booleans
var result = is_positive.select(values * 2, values * -1)
print(result)  # => [2, 2, 6, 4]
```

**Horizontal reductions:**

```mojo
# Sum all elements in a vector
var data = SIMD[DType.float64, 4](10.5, 20.3, 30.1, 40.7)
var total = data.reduce_add()
var maximum = data.reduce_max()
print(total, maximum)  # => 101.6 40.7
```

## Constraints

The size of the SIMD vector must be positive and a power of 2.

## Parameters

- **dtype** (`DType`): The data type of SIMD vector elements.
- **size** (`Int`): The size of the SIMD vector (number of elements).

---

## Compile-time members

- `MAX` — Maximum value for the SIMD value, potentially `+inf`.
- `MAX_FINITE` — Maximum finite value of SIMD value.
- `MIN` — Minimum value for the SIMD value, potentially `-inf`.
- `MIN_FINITE` — Minimum (lowest) finite value of SIMD value.
- `device_type` — SIMD types are remapped to the same type when passed to accelerator devices.

---

## Construction

### From another SIMD (with casting)

```mojo
__init__[other_dtype: DType, //](value: SIMD[other_dtype, size], /) -> Self
```

Initialize from another SIMD of the same size. If the value passed is a scalar, you can initialize a SIMD vector with more elements.

```mojo
print(UInt64(UInt8(42))) # 42
print(SIMD[DType.uint64, 4](UInt8(42))) # [42, 42, 42, 42]
```

**Casting behavior:**

```mojo
# Basic casting preserves value within range
Int8(UInt8(127)) == Int8(127)

# Numbers above signed max wrap to negative using two's complement
Int8(UInt8(128)) == Int8(-128)
Int8(UInt8(129)) == Int8(-127)
Int8(UInt8(256)) == Int8(0)

# Negative signed cast to unsigned using two's complement
UInt8(Int8(-128)) == UInt8(128)
UInt8(Int8(-127)) == UInt8(129)
UInt8(Int8(-1)) == UInt8(255)

# Truncate precision after downcast and upcast
Float64(Float32(Float64(123456789.123456789))) == Float64(123456792.0)

# Rightmost bits of significand become 0's on upcast
Float64(Float32(0.3)) == Float64(0.30000001192092896)

# Float to int/uint floors
Int64(Float64(42.2)) == Int64(42)
```

### Splat constructors

```mojo
__init__(value: Int, /) -> Self
```
Splats a signed integer across all elements.

```mojo
__init__(value: Scalar[dtype], /) -> Self
```
Splats a scalar value across all elements.

```mojo
__init__(*, fill: Bool) -> SIMD[DType.bool, size]
```
Splats a bool value across all elements of a bool SIMD vector.

### Variadic element constructor

```mojo
__init__(*elems: Scalar[dtype], ...) -> Self
```
Constructs a SIMD vector from a variadic list of elements. The number of input values must equal `size`.

### Bit-level construction

```mojo
__init__[int_dtype: DType, //](*, from_bits: SIMD[int_dtype, size]) -> Self
```
Initializes the SIMD vector from the bits of an integral SIMD vector.

### Default

```mojo
__init__() -> Self
```
SIMD vectors are default-initialized to all zeros.

---

## Element access

```mojo
__getitem__(self, idx: Int) -> Scalar[dtype]
__setitem__(mut self, idx: Int, val: Scalar[dtype])
__len__(self) -> Int
__contains__(self, value: Scalar[dtype]) -> Bool
```

---

## Arithmetic operators

Standard element-wise arithmetic. Each returns a new vector where element `i` is computed from `self[i]` and `rhs[i]`.

| Operator | Method | Description |
|---|---|---|
| `+` | `__add__` | Element-wise addition |
| `-` | `__sub__` | Element-wise subtraction |
| `*` | `__mul__` | Element-wise multiplication |
| `/` | `__truediv__` | Element-wise division |
| `//` | `__floordiv__` | Floor division (numeric types) |
| `%` | `__mod__` | Element-wise remainder |
| `**` | `__pow__` | Element-wise power (by `Int` or `Self`) |
| unary `-` | `__neg__` | Negation |
| unary `+` | `__pos__` | Identity |

In-place versions (`+=`, `-=`, `*=`, `/=`, `//=`, `%=`, `**=`) and reflected versions (`__radd__`, etc.) are all available.

### `__divmod__`

```mojo
__divmod__(self, denominator: Self) -> Tuple[SIMD[dtype, size], SIMD[dtype, size]]
```
Returns `(self // denominator, self % denominator)`.

### `__ceildiv__`

```mojo
__ceildiv__(self, denominator: Self) -> Self
```
Rounded-up division of `self` by `denominator`.

---

## Bitwise operators

Available for boolean or integral element types.

| Operator | Method | Description |
|---|---|---|
| `&` | `__and__` | Bitwise AND |
| `\|` | `__or__` | Bitwise OR |
| `^` | `__xor__` | Bitwise XOR |
| `~` | `__invert__` | Bitwise NOT |
| `<<` | `__lshift__` | Left shift (integral only) |
| `>>` | `__rshift__` | Right shift (integral only) |

In-place (`&=`, `|=`, `^=`, `<<=`, `>>=`) and reflected forms are available.

---

## Comparison

Scalar comparisons (return `Bool`, require scalar SIMD): `__lt__`, `__le__`, `__eq__`, `__ne__`, `__gt__`, `__ge__`.

**Element-wise comparisons** (return `SIMD[DType.bool, size]`):

```mojo
eq(self, rhs: Self) -> SIMD[DType.bool, size]
ne(self, rhs: Self) -> SIMD[DType.bool, size]
lt(self, rhs: Self) -> SIMD[DType.bool, size]
le(self, rhs: Self) -> SIMD[DType.bool, size]
gt(self, rhs: Self) -> SIMD[DType.bool, size]
ge(self, rhs: Self) -> SIMD[DType.bool, size]
```

Each produces a bool SIMD vector where element `i` is the result of comparing `self[i]` and `rhs[i]`.

---

## Math operations

### `__abs__`, `__neg__`, `__floor__`, `__ceil__`, `__trunc__`, `__round__`

Element-wise absolute value, negation, floor, ceiling, truncation, and rounding. Rounding uses banker's rounding (IEEE 754 default). `__round__` also accepts an `ndigits: Int` argument.

### `fma`

```mojo
fma[flag: FastMathFlag = FastMathFlag.CONTRACT](
    self, multiplier: Self, accumulator: Self
) -> Self
```

Performs a fused multiply-add: `self * multiplier + accumulator`. Element `i` is `self[i]*multiplier[i] + accumulator[i]`.

### `clamp`

```mojo
clamp(self, lower_bound: Self, upper_bound: Self) -> Self
```

Clamps values to `[lower_bound, upper_bound]`. For example, `[0, 1, 2, 3]` clamped to `[1, 2]` gives `[1, 1, 2, 2]`.

### `is_power_of_two`

```mojo
is_power_of_two(self) -> SIMD[DType.bool, size]
```

Element-wise check if each element is a power of 2. Integral element type only.

---

## Type conversion

### `cast`

```mojo
cast[target: DType](self) -> SIMD[target, size]
```

Casts elements to the target element type. See the casting behavior examples under construction — downcasts wrap via two's complement, upcasts may introduce precision artifacts, float-to-int floors.

### `to_bits`

```mojo
to_bits[_dtype: DType = ...](self) -> SIMD[_dtype, size]
```

Bitcasts the SIMD vector to an integer SIMD vector. Useful for inspecting the bit pattern of floating-point values.

### `from_bytes` / `as_bytes`

```mojo
static from_bytes[*, big_endian: Bool = is_big_endian()](
    bytes: InlineArray[Byte, size_of[SIMD[dtype, size]]()]
) -> Self

as_bytes[*, big_endian: Bool = is_big_endian()](self) 
    -> InlineArray[Byte, size_of[SIMD[dtype, size]]()]
```

Convert to/from a byte array with configurable endianness.

### Scalar conversions

- `__bool__(self) -> Bool` — non-zero test (scalar).
- `__int__(self) -> Int` — truncating cast to `Int` (scalar only).
- `__float__(self) -> Float64` — cast to `Float64` (scalar only).

---

## Vector manipulation

### `shuffle`

```mojo
shuffle[*mask: Int](self) -> Self
shuffle[*mask: Int](self, other: Self) -> Self
shuffle[mask: IndexList[size, ...]](self) -> Self
shuffle[mask: IndexList[size, ...]](self, other: Self) -> Self
```

Permutes (blends) vector values using a mask. For the two-vector form, mask values index into the concatenation of `self` and `other`, so valid values are in `[0, 2*size)`. The result has the same length as the mask, with position `i` set to `(self + other)[mask[i]]`.

### `slice`

```mojo
slice[output_width: Int, /, *, offset: Int = 0](self) -> SIMD[dtype, output_width]
```

Returns a sub-vector of width `output_width` starting at `offset`. Requires `output_width + offset <= size`.

### `insert`

```mojo
insert[*, offset: Int = 0](self, value: SIMD[dtype, value.size]) -> Self
```

Returns a new vector with elements `[offset : offset + value.size]` replaced by `value`. `offset` must be a multiple of `value.size`.

### `join`

```mojo
join(self, other: Self) -> SIMD[dtype, (2 * size)]
```

Concatenates: `[self_0, ..., self_n, other_0, ..., other_n]`.

### `interleave`

```mojo
interleave(self, other: Self) -> SIMD[dtype, (2 * size)]
```

Interleaves: `[self_0, other_0, self_1, other_1, ..., self_n, other_n]`.

### `split`

```mojo
split(self) -> Tuple[SIMD[dtype, (size // 2)], SIMD[dtype, (size // 2)]]
```

Splits into two equal halves.

### `deinterleave`

```mojo
deinterleave(self) -> Tuple[SIMD[dtype, (size // 2)], SIMD[dtype, (size // 2)]]
```

Returns `(evens, odds)` — evens are `self_0, self_2, ...`, odds are `self_1, self_3, ...`. Requires `size > 1`.

### `reversed`

```mojo
reversed(self) -> Self
```

Reverses the vector by index.

```mojo
print(SIMD[DType.uint8, 4](1, 2, 3, 4).reversed()) # [4, 3, 2, 1]
```

### `rotate_left` / `rotate_right`

```mojo
rotate_left[shift: Int](self) -> Self   # -size <= shift < size
rotate_right[shift: Int](self) -> Self  # -size < shift <= size
```

Rotates elements with wrap-around.

### `shift_left` / `shift_right`

```mojo
shift_left[shift: Int](self) -> Self    # 0 <= shift <= size
shift_right[shift: Int](self) -> Self   # 0 <= shift <= size
```

Shifts elements without wrap-around, filling with zero.

---

## Reductions (horizontal)

All reductions take an optional `size_out: Int = 1` parameter. `size_out` must not exceed the vector width.

### `reduce`

```mojo
reduce[func: ..., size_out: Int = 1](self) -> SIMD[dtype, size_out]
```

Reduces using a user-provided binary reduction function (both capturing and non-capturing variants).

### Specialized reductions

| Method | Description | Constraints |
|---|---|---|
| `reduce_add` | Sum of elements | — |
| `reduce_mul` | Product of elements | integer or FP |
| `reduce_min` | Minimum element | integer or FP |
| `reduce_max` | Maximum element | integer or FP |
| `reduce_and` | Bitwise AND across elements | integer or boolean |
| `reduce_or`  | Bitwise OR across elements | integer or boolean |
| `reduce_bit_count` | Total set bits across all elements (returns `Int`) | integral or boolean |

---

## Conditional selection

### `select`

```mojo
select[_dtype: DType](
    self,
    true_case: SIMD[_dtype, size],
    false_case: SIMD[_dtype, size],
) -> SIMD[_dtype, size]
```

Called on a **boolean** SIMD mask. Returns a new vector where element `i` is `true_case[i]` if `self[i]` is `True`, otherwise `false_case[i]`.

---

## Implemented traits

`Absable`, `AnyType`, `Boolable`, `CeilDivable`, `Ceilable`, `Comparable`, `ConvertibleToPython`, `Copyable`, `Defaultable`, `DevicePassable`, `DivModable`, `Equatable`, `Floorable`, `Hashable`, `ImplicitlyCopyable`, `ImplicitlyDestructible`, `Indexer`, `Intable`, `Movable`, `Powable`, `RegisterPassable`, `Roundable`, `Sized`, `TrivialRegisterPassable`, `Truncable`, `Writable`, `_FromInt`.
