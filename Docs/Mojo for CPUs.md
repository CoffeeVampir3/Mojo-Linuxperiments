# Mojo Language Reference

Keynotes:
NOT RUST
NOT PYTHON
NOT C++
NOT GO

## 1. Core Language Features

### Variables & Constants
```mojo
x = 3
var y: Int = 3
comptime N = 1024
```

### Functions
```mojo
# def is raising by default
def might_fail(x: Int):
    if x == 0:
        raise "invalid input"

# fn must opt-in with `raises`
fn safe_use() raises -> None:
    try:
        might_fail(0)
    except e:
        print("Error:", e)

# Typed errors - specify error type after raises
fn foo() raises CustomError -> Int:
    raise CustomError("failed")

fn caller():
    try:
        print(foo())
    except err:  # err is typed as CustomError
        print(err)

# Never type - for functions that never return normally
fn abort_now() -> Never:
    abort()

# Never as an error type means the function can't raise.
fn doesnt_raise() raises Never -> Int:
    return 123

# Named results (out) vs -> return
fn incr(a: Int) -> Int:
    return a + 1

fn incr(a: Int, out b: Int):
    b = a + 1  # equivalent
```

### Error Handling: def vs fn

- `def` functions raise by default (Python compatibility)
- `fn` functions must declare `raises` explicitly
- Calling a `def` from `fn` requires `raises` or `try/except`

This is why stdlib buffer functions declare `raises` - they may call Python-style code.

### Function Type Conversions

Implicit conversions allowed between function types:
- Non-raising to raising function
- Functions whose result types are implicitly convertible

```mojo
fn takes_raising_float(a: fn () raises -> Float32): ...
fn returns_int() -> Int: ...
fn example():
    takes_raising_float(returns_int)  # Valid: Int -> Float32, non-raising -> raising
```

### Lifetimes, Origins, and References

The Mojo compiler includes a **lifetime checker**, a compiler pass that analyzes dataflow through your program. It identifies when variables are valid and inserts destructor calls when a variable's lifetime ends (**ASAP destruction**).

The compiler uses a special value called an **origin** to track the lifetime of variables and the validity of references. An origin answers two questions:
1. What variable "owns" this value?
2. Can the value be mutated using this reference?

Origin tracking and lifetime checking is done at **compile time**. Origins track variables symbolically, allowing the compiler to identify lifetimes and ensure references remain valid.

```mojo
def print_str(s: String):
    print(s)

def main():
    name: String = "Joan"
    print_str(name)  # s gets immutable reference to name's storage
```

**When you need origins explicitly:**
- `ref` arguments and `ref` return values
- Types like `Pointer` or `Span` parameterized on origin

---

## Origin Types

| Type | Description |
|------|-------------|
| `Origin` | Origin token (comptime value) |
| `ImmutOrigin` | Immutable origin (comptime value) |
| `MutOrigin` | Mutable origin (comptime value) |

```mojo
struct ImmutRef[origin: ImmutOrigin]:
    pass

struct ParametricRef[origin: Origin]:
    pass

# Origin conversion
comptime o: MutOrigin = MutOrigin.external
comptime immut: ImmutOrigin = ImmutOrigin(o)            # Safe: drop mutability
comptime mut: MutOrigin = MutOrigin(unsafe_cast=immut)  # Unsafe: add mutability

from memory import Pointer
def use_pointer():
    a = 10
    ptr = Pointer(to=a)  # origin inferred from a
```

**OriginSet**: Represents a group of origins for tracking lifetimes of values captured in closures.

---

## Origin Values

| Origin Value | Description |
|--------------|-------------|
| `StaticConstantOrigin` | Immutable values lasting program duration (e.g., string literals) |
| `origin_of(value)` | Derived origin from value(s) |
| Inferred | Captured from function argument via parameter inference |
| `MutOrigin.external` / `ImmutOrigin.external` | Untracked memory (e.g., dynamic allocation) |
| `MutAnyOrigin` / `ImmutAnyOrigin` | Wildcard - might access any live value (disables ASAP destruction) |

```mojo
origin_of(self)
origin_of(x.y)
origin_of(foo())      # analyzed statically, foo() not called
origin_of(a, b)       # union of origins

from memory import OwnedPointer, Pointer

struct BoxedString:
    var o_ptr: OwnedPointer[String]

    fn __init__(out self, value: String):
        self.o_ptr = OwnedPointer(value)

    fn as_ptr(mut self) -> Pointer[String, origin_of(self.o_ptr)]:
        return Pointer(to=self.o_ptr[])
```

**Origin unions**: Union of origins extends all constituent lifetimes. Mutable only if all constituents are mutable.

**External origins**: For memory not owned by any variable (e.g., `alloc()` returns `MutOrigin.external`). You manage the lifetime.

**Wildcard origins**: Discouraged. Using a wildcard-origin pointer disables ASAP destruction for all values in scope while the pointer is live.

---

## ref Arguments

Parametric mutability: accept mutable or immutable references without knowing in advance.

```mojo
ref arg_name: arg_type                    # origin/mutability inferred
ref [origin_specifier] arg_name: arg_type # explicit origin

def add_ref(ref a: Int, b: Int) -> Int:
    return a + b
```

**Origin specifiers:**
- Origin value
- Expression (shorthand for `origin_of(expression)`)
- AddressSpace value
- `_` (unbound/infer)

```mojo
from collections import List
from memory import Span

def to_byte_span[
    origin: Origin,
](ref [origin] list: List[Byte]) -> Span[Byte, origin]:
    return Span(list)

def main():
    list: List[Byte] = [77, 111, 106, 111]
    span = to_byte_span(list)  # origin inferred, span lifetime tied to list
```

---

## ref Return Values

Return a reference (not a copy) with explicit origin.

```mojo
-> ref [origin_specifier] arg_type
```

```mojo
struct NameList:
    var names: List[String]

    def __init__(out self, *names: String):
        self.names = []
        for name in names:
            self.names.append(name)

    def __getitem__(ref self, index: Int) ->
        ref [self.names] String:
        if (index >= 0 and index < len(self.names)):
            return self.names[index]
        else:
            raise Error("index out of bounds")

def main():
    list = NameList("Thor", "Athena", "Dana", "Vrinda")
    ref name = list[2]     # reference binding
    name += "?"
    print(list[2])         # Dana?
```

**Assignment vs binding:**
```mojo
var name_copy = list[2]  # owned copy
ref name_ref = list[2]   # reference to list[2]
```

**Parametric mutability**: Return value mutability follows `self` mutability:
```mojo
fn pass_immutable_list(list: NameList) raises:
    print(list[2])
    # list[2] += "?"  # Error: immutable
```

**Union origins**: Return from multiple sources:
```mojo
def pick_one(cond: Bool, ref a: String, ref b: String) -> ref [a, b] String:
    return a if cond else b
```

---

## Lifecycle & Ownership

**Core Rules:**
1. Each value has exactly one owner
2. Value destroyed when owner's lifetime ends (ASAP destruction)
3. Lifetime extended if references exist

**Variables own values. Structs own fields. References access values owned elsewhere.**

Reference bindings: `ref value_ref = list[0]`

---

## Argument Conventions

| Convention | Ownership | Mutability | Description |
|------------|-----------|------------|-------------|
| `read` | Callee borrows | Immutable | Immutable reference (default) |
| `mut` | Callee borrows | Mutable | Mutable reference |
| `var` | Callee owns | Mutable | Ownership transfer or copy |
| `ref` | Callee borrows | Parametric | Generalized read/mut (advanced) |
| `out` | Special | N/A | Uninitialized → must initialize before return |
| `deinit` | Special | N/A | Initialized → uninitialized at return (any arg in struct methods) |

---

## `read` (Default)

- Immutable reference, no copy
- `@register_passable` types (Int, Float, SIMD) pass in registers, not by indirection
- No default values allowed for any convention except read

## `mut`

- Mutable reference, changes visible to caller
- Caller must pass mutable variable
- Cannot form mutable ref from immutable ref
- **Exclusivity enforced**: No other references (mutable or immutable) to same value allowed
- Exclusivity not enforced for register-passable trivial types (they copy)
- No default values allowed

## `var` with `^` Transfer Sigil

**Three transfer modes:**

1. **With `^`**: Ends caller variable lifetime, transfers ownership
2. **Without `^`**: Copies value (requires `__copyinit__()`), caller retains ownership
3. **Rvalue**: Direct transfer of newly-created values (no variable owns it)

**Destruction**: `var` value destroyed at function exit unless transferred elsewhere (e.g., `list.append(name^)`)

---

## Transfer Implementation

Ownership transfer ≠ guaranteed move operation. Three mechanisms:

1. `__moveinit__()` if implemented
2. `__copyinit__()` then destroy original if no `__moveinit__()`
3. Optimization: ownership update without constructor invocation

**Requirement**: Type must have `__copyinit__()` for `var` without `^`

---

## Key Constraints

- Cannot pass same value as both `mut` and any other reference (exclusivity)
- Cannot use variable after `^` transfer (compile error)
- `mut` arguments must receive mutable variables
- Lifetime checker prevents use-after-free, double-free, memory leaks

### SIMD Operations

```mojo
# Strided: extract every Nth element (e.g., R from RGB with stride=3)
vals = ptr.offset(i).strided_load[width=8](stride)
ptr.offset(i).strided_store[width=8](vals, stride)

# Gather/scatter: load/store from vector of offsets
vals = ptr.gather[width=8](offsets)
ptr.scatter[width=8](vals, offsets)
```

## Safety Notes

- No bounds checking on arithmetic
- Nullable by default
- Manual memory management
- Origin system tracks lifetimes automatically
- Freeing same memory twice = UB
- Double-check: who allocates, who frees

**Linear types support**: `UnsafePointer`, `Pointer`, `Variant`, and `VariadicPack` can contain linear types (non-implicitly-destructible types).

```mojo
fn __init__(out self, args)
fn __del__(deinit self)
fn __copyinit__(out self, existing: Self)
fn __moveinit__(out self, deinit existing: Self)

@fieldwise_init
struct MyType(Copyable, ImplicitlyCopyable)

@register_passable("trivial")
struct TrivialType

fn read_only(val: Int)
fn mutable(mut val: Int)
fn take_ownership(var val: Int)
transfer(val^)

fn with_origin[origin: Origin](ref [origin] data: Type) -> ref [origin] Type
origin_of(value) | origin_of(a, b)

ref item_ref = list[0]
item_ref += 1
```

### Operators
```mojo
fn __pos/neg/invert__(self) -> Self
fn __add/radd/iadd__(self|mut self, rhs: Self) -> Self|void
fn __eq/ne/lt/le/gt/ge__(self, other: Self) -> Bool
result = simd1.eq/ne/lt/le/gt/ge(simd2)
fn __getitem__(self, idx: Int) -> T
fn __setitem__(mut self, idx: Int, val: T)

quotient, remainder = divmod(simd_a, simd_b)
```

### Parameters (Compile-Time Metaprogramming)

Parameters are compile-time values that become runtime constants. In Mojo, "parameter" = compile-time, "argument" = runtime.

```mojo
# Parameterized functions
fn repeat[count: Int](msg: String):
    @parameter                             # Compile-time loop unrolling
    for i in range(count):
        print(msg)

repeat[3]("Hello")                         # Compiler creates concrete version

# Parameter list anatomy
fn example[
    dtype: DType,                          # Infer-only (before //)
    width: Int,
    //,
    values: SIMD[dtype, width],            # Positional-only (before /)
    /,
    compare: fn(Scalar[dtype], Scalar[dtype]) -> Int,  # Positional-or-keyword
    *,
    reverse: Bool = False,                 # Keyword-only (after *)
]():
    pass

# Parameter inference
fn rsqrt[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return 1 / sqrt(x)

rsqrt(Float16(42))                         # dt inferred from argument type

# Infer-only parameters (before //)
fn dependent[dtype: DType, //, value: Scalar[dtype]]():
    print(value)

dependent[Float64(2.2)]()                  # dtype inferred, value specified

# Variadic parameters
struct MyTensor[*dimensions: Int]:
    pass

fn sum_params[*values: Int]() -> Int:
    comptime list = VariadicList(values)
    var sum = 0
    for v in list:
        sum += v
    return sum

# Optional and keyword parameters
fn speak[a: Int = 3, msg: String = "woof"]():
    print(msg, a)

speak()                                    # woof 3
speak[5]()                                 # woof 5
speak[msg="meow"]()                        # meow 3
```

**Parameterized Structs:**
```mojo
struct GenericArray[ElementType: Copyable]:
    var data: UnsafePointer[Self.ElementType]
    var size: Int

    fn __getitem__(self, i: Int) -> ref [self] Self.ElementType:
        return self.data[i]

var arr: GenericArray[Int] = [1, 2, 3]

# Accessing struct parameters
print(SIMD[DType.float32, 4].size)         # On type: 4
var x = SIMD[DType.int32, 2](4, 8)
print(x.dtype)                             # On instance: int32

# Conditional conformance
struct Container[ElementType: Movable]:
    var element: Self.ElementType

    fn __str__[T: Writable & Movable, //](self: Container[T]) -> String:
        return String(self.element)        # Only works if ElementType is Writable
```

**comptime Declarations:**
```mojo
# Named compile-time constants
comptime rows = 512
comptime block_size = _calculate_block_size()

# Force a subexpression to be evaluated at compile time
fn takes_layout[a: Layout]():
    print(comptime(a.size()))

# Type aliases
comptime Float16 = SIMD[DType.float16, 1]
comptime UInt8 = SIMD[DType.uint8, 1]

# Parametric comptime values
comptime AddOne[a: Int]: Int = a + 1
comptime nine = AddOne[8]

# Parametric type aliases
comptime TwoOfAKind[dt: DType] = SIMD[dt, 2]
comptime StringKeyDict[V: Copyable] = Dict[String, V]

var floats = TwoOfAKind[DType.float32](1.0, 2.0)
var dict: StringKeyDict[Int] = {"answer": 42}

# comptime struct members
struct Circle[radius: Float64]:
    comptime pi = 3.14159265359
    comptime circumference = 2 * Self.pi * Self.radius

# comptime as enum pattern
struct Sentiment(Equatable):
    var _value: Int
    comptime NEGATIVE = Sentiment(0)
    comptime NEUTRAL = Sentiment(1)
    comptime POSITIVE = Sentiment(2)
```

**Automatic Parameterization:**
```mojo
# Unbound type = auto-parameterized function
fn print_info(vec: SIMD):                  # SIMD[*_] - all params unbound
    print(vec.dtype, vec.size)

# Equivalent to:
fn print_info[dt: DType, sz: Int, //](vec: SIMD[dt, sz]):
    print(vec.dtype, vec.size)

# Partially-bound types
fn eat(f: Fudge[5, *_]):                   # sugar=5, others unbound
    pass

fn devour(f: Fudge[_, 6, _]):              # cream=6, others unbound
    pass

# Using type_of for matching
fn interleave(v1: SIMD, v2: type_of(v1)) -> SIMD[v1.dtype, v1.size * 2]:
    pass
```

**Bound/Unbound Types:**
```mojo
# Fully bound (concrete, instantiable)
var x: SIMD[DType.float32, 4]

# Partially bound
comptime StringDict = Dict[String, _]      # Key bound, Value unbound
var d: StringDict[Int] = {}

# Unbound patterns
MyType[*_]                                 # All positional params unbound
MyType[**_]                                # All keyword params unbound
MyType[_, _, _]                            # Explicit individual unbinding

# Partially bound in signatures
fn foo(m: MyType["Hello", _, _, True]):    # Some bound, some unbound
    pass
```

**Compile-Time Control Flow:**
```mojo
# @parameter if - compile-time branching
fn reduce_add(x: SIMD) -> Int:
    @parameter
    if x.size == 1:
        return Int(x[0])
    elif x.size == 2:
        return Int(x[0]) + Int(x[1])
    comptime half = x.size // 2
    return reduce_add(slice(x, 0) + slice(x, half))

# @parameter for - compile-time loop unrolling
@parameter
for i in range(4):                         # Must have compile-time bounds
    process[i]()
```

**rebind() for Type Coercion:**
```mojo
fn take_simd8(x: SIMD[DType.float32, 8]):
    pass

fn generic[nelts: Int](x: SIMD[DType.float32, nelts]):
    @parameter
    if nelts == 8:
        take_simd8(rebind[SIMD[DType.float32, 8]](x))  # Assert types match
```

**where Clauses (Experimental):**
```mojo
# DType constraints
fn foo[dt: DType]() -> Int where dt is DType.int32:
    return 42

# DType predicates: is_signed(), is_unsigned(), is_numeric(), is_integral(),
#                   is_floating_point(), is_float8(), is_half_float()

# SIMD constraints
fn bar[dt: DType, x: Int]() -> Int where SIMD[dt, 4](x) + 2 > SIMD[dt, 4](0):
    return 42
```

### Structs
```mojo
@fieldwise_init
struct MyStruct(Copyable):
    var field1: Int
    var field2: String

    fn method(self) -> Result
    fn mutating(mut self)

    @staticmethod
    fn static_method(args)

    fn __init__(out self, args)
    fn __del__(deinit self)
    fn __getitem/setitem/len/str/repr__(self)
    fn write_to(self, mut writer: Writer)

# Context managers (with statements)
struct MyContextManager:
    fn __enter__(self): ...
    fn __exit__(self): ...                           # Normal exit
    fn __exit__[E: AnyType](self, err: E) -> Bool: ...  # Error exit (typed)

# Consuming context managers (linear types)
struct ConsumingCtxMgr:
    fn __enter__(self): ...
    fn __exit__(var self): ...                       # Consumes self on exit
    fn __exit__(deinit self): ...                    # Also valid
```

### Traits

Traits define a contract: a set of requirements a type must implement. Similar to Java interfaces, C++ concepts, Swift protocols, and Rust traits.

```mojo
# Defining traits
trait Quackable:
    fn quack(self): ...                    # Required (no default implementation)

trait DefaultQuackable:
    fn quack(self): pass                   # Default do-nothing implementation

trait WithBody:
    fn greet(self):                        # Default implementation with body
        print("Hello")

trait HasStatic:
    @staticmethod
    fn do_stuff(): ...                     # Static methods supported

# Conforming to traits
@fieldwise_init
struct Duck(Copyable, Quackable):
    fn quack(self):
        print("Quack")

@fieldwise_init
struct DefaultDuck(Copyable, DefaultQuackable):
    pass                                   # Inherits default quack()

# Using traits as type bounds
fn make_quack[T: Quackable](duck: T):
    duck.quack()

fn make_quack(duck: Some[Quackable]):      # Shorthand form
    duck.quack()

fn take_two[T: Quackable](a: T, b: T):     # Same type constraint
    pass

# Trait composition with &
comptime QuackFly = Quackable & Flyable

fn needs_both[T: Quackable & Flyable](x: T): pass
fn needs_both(x: Some[Quackable & Flyable]): pass

struct FlyingDuck(Quackable, Flyable):     # Conforms to QuackFly
    fn quack(self): pass
    fn fly(self): pass

# Trait inheritance
trait Animal:
    fn make_sound(self): ...

trait Bird(Animal):                        # Bird requires Animal methods too
    fn fly(self): ...

trait Named:
    fn get_name(self) -> String: ...

trait NamedAnimal(Animal, Named):          # Multiple inheritance
    pass

# comptime members for generics
trait Stacklike:
    comptime EltType: Copyable             # Required type member

    fn push(mut self, var item: Self.EltType): ...
    fn pop(mut self) -> Self.EltType: ...

struct MyStack[type: Copyable](Stacklike):
    comptime EltType = Self.type           # Concrete type assignment
    var list: List[Self.EltType]

    fn push(mut self, var item: Self.EltType):
        self.list.append(item^)

    fn pop(mut self) -> Self.EltType:
        return self.list.pop()

# Lifecycle traits
comptime MassProducible = Defaultable & Movable

fn factory[T: MassProducible]() -> T:
    return T()

# Register-passable traits
@register_passable
trait TrivialTrait:                        # Conformers must be @register_passable
    pass

@register_passable("trivial")
trait VeryTrivial:                         # Conformers must be @register_passable("trivial")
    pass
```

**Built-in Traits:**
| Trait | Requires |
|-------|----------|
| `Sized` | `__len__(self) -> Int` |
| `Intable` | `__int__(self) -> Int` |
| `IntableRaising` | `__int__(self) raises -> Int` |
| `Stringable` | `__str__(self) -> String` |
| `StringableRaising` | `__str__(self) raises -> String` |
| `Representable` | `__repr__(self) -> String` |
| `Writable` | `write_to(self, mut writer: Some[Writer])` |
| `Boolable` | `__bool__(self) -> Bool` |
| `Hashable` | `__hash__(self) -> UInt` |
| `Equatable` | `__eq__(self, other: Self) -> Bool` |
| `Comparable` | `__lt__`, `__le__`, `__gt__`, `__ge__` |
| `Movable` | `__moveinit__(out self, deinit existing: Self)` |
| `Copyable` | `__copyinit__(out self, existing: Self)` (refines `Movable`) |
| `Defaultable` | `__init__(out self)` |
| `AnyType` | Base trait, no `__del__()` required (supports linear types) |
| `ImplicitlyDestructible` | `__del__()` callable by compiler (use for generic destructible types) |
| `KeyElement` | `Copyable & Hashable & Equatable` |

```mojo
# Writable + Stringable + Representable pattern
@fieldwise_init
struct Dog(Copyable, Stringable, Representable, Writable):
    var name: String
    var age: Int

    fn write_to(self, mut writer: Some[Writer]):
        writer.write("Dog(", self.name, ", ", self.age, ")")

    fn __str__(self) -> String:
        return String.write(self)

    fn __repr__(self) -> String:
        return String("Dog(name=", repr(self.name), ", age=", repr(self.age), ")")

# Compile-time trait conformance check (experimental)
fn maybe_print[T: AnyType](value: T):
    @parameter
    if conforms_to(T, Writable):
        print(trait_downcast[Writable](value))
    else:
        print("[UNPRINTABLE]")
```

## 2. Data Types

### Numeric Types
```mojo
Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Int, UInt
Float16, Float32, Float64
DType.float4_e2m1fn

vec = SIMD[DType.f32, 4](1.0, 2.0, 3.0, 4.0)
result = vec1 + vec2 | vec * scalar
result = Float32(int_value) | simd1.cast[DType.i32]()

0xFF, 0o77, 0b1010
3.14, 1.2e9
```

**No implicit conversion between `Int` and `UInt`** - explicit casts required.

### Complex Numbers
```mojo
c = ComplexSIMD[dtype, size](re, im)
c = ComplexSIMD(*, from_interleaved: SIMD)
c = ComplexSIMD(*, from_deinterleaved: SIMD)

c.fma(b, c) | squared_add(c) | norm() | squared_norm() | conj()
abs(c)
```

### Tuples
```mojo
t = (1, "a", 3.14)
(1, "a") == (1, "a")
(1, "a") < (1, "b")
(1, "a") <= (2, "z")
t.concat(other_tuple)
t.reverse()
```

### Strings
```mojo
s = "Hello" + " World"
s *= 3
if "sub" in s: pass
char = s[0]
substring = s[start:end:step]

comptime CONST = "compile time"
comptime MULTI = """multiline"""

# String Methods
String() | String(capacity=1024) | String(unsafe_uninit_length=n)
String(bytes=span) | String(unsafe_from_utf8_ptr=ptr)

str.capacity() -> Int | byte_length()
len(str.codepoints())
str.unsafe_ptr() | unsafe_ptr_mut(capacity) | unsafe_cstr_ptr() | as_bytes()
str.reserve(capacity: Int) | resize(length, fill_byte) | resize(unsafe_uninit_length=n)

str += other
str.join(list) | split(sep) | strip() | lower() | upper()
str.replace(old, new) | find(substr, start) | count(substr)
str.startswith(prefix, start, end) | format(*args)

# StringLiteral.format emits compile-time constraint error for invalid format strings
"Hello, {!invalid}".format("world")  # Compile error: Conversion flag "invalid" not recognized

atol(str), atof(str), chr(codepoint), ord(str)

# StringSlice
slice = str.as_string_slice()
slice.codepoints() | codepoint_slices()
slice[idx] | slice[start:end]

slice = StringSlice("literal")
slice = StringSlice(*, unsafe_from_utf8: Span[UInt8])
slice = StringSlice(*, from_utf8: Span[UInt8])
slice = StringSlice(*, ptr: UnsafePointer[UInt8], length: Int)

slice[0:5]
slice.byte_length() | char_length() | is_codepoint_boundary(index)
slice.split(sep) | strip() | codepoints() | as_bytes()

# CStringSlice - nul-terminated C-style strings
cslice = CStringSlice(ptr)
cslice.byte_length() | as_string_slice()

# Codepoint
cp = Codepoint.from_u32(val) | ord("a") | Codepoint(unsafe_unchecked_codepoint=val)
cp.to_u32() | utf8_byte_length() -> Int | unsafe_write_utf8[optimize_ascii=True](ptr) -> Int
Codepoint.unsafe_decode_utf8_codepoint(span)
cp.is_ascii() | is_ascii_digit() | is_ascii_upper() | is_ascii_lower()
cp.is_python_space() | is_posix_space()
cp1 < cp2 | cp1 == cp2 | cp1 <= cp2
```

### Collections
```mojo
from collections import Set, Deque, Counter, LinkedList, BitSet
from collections.interval import Interval

# List (conforms to Equatable, Writable, Stringable, Representable)
var list: List[Int] = [1, 2, 3]
list = List[Int]() | List[Int](capacity=1024) | List[Int](length: Int=10, fill=0)
list.append(4) | insert(idx, val) | pop() | pop(idx)
list.reserve(capacity) | resize(new_size, value) | resize(unsafe_uninit_length=n)
list.unsafe_get(idx) | unsafe_set(idx, val) | unsafe_ptr()
for ref item in list: item += 1

# Span (conforms to Iterable)
span = Span[T](ptr, length: Int)
span.binary_search_by[comparator](value)
span.unsafe_get(idx) | unsafe_swap_elements(i, j)
subspan = span.unsafe_subspan(offset, length)
for item in span: pass

# Dict (raises DictKeyError on missing key; conforms to Writable, Stringable, Representable)
dict = Dict[String, Int]() | Dict[String, Int](power_of_two_initial_capacity=1024)
dict = {"key": value}
dict[key] = value
dict.get(key) | get(key, default)
dict.pop(key) | pop(key, default)
dict.keys() | values() | items() | update(other)
dict | other

# Set
set = Set[Int](1, 2, 3) | {1, 2, 3}
set.add(val) | remove(val) | discard(val) | pop()
val in set
set1 | set2  # union
set1 & set2  # intersection
set1 - set2  # difference
set1 ^ set2  # symmetric difference
set1 < set2 | set1 <= set2  # subset

# Deque
deque = Deque[Int](capacity=64) | Deque[Int](maxlen=100)
deque.append(val) | appendleft(val) | pop() | popleft() | insert(idx, val)
deque[idx]
deque.rotate(n)

# Counter
counter = Counter[String]("a", "a", "b")
counter[key]
counter.most_common(n) | total()
counter1 + counter2
counter1 - counter2
counter1 & counter2
counter1 | counter2

# LinkedList
list = LinkedList[Int](1, 2, 3)
list.append(val) | prepend(val) | pop() | pop(idx)
list[idx]
list.reverse() | insert(idx, val)

# InlineArray
arr = InlineArray[Int, 3](1, 2, 3)
arr = InlineArray[Int, 5](fill=42)
arr = InlineArray[Int, 10](uninitialized=True)
arr.unsafe_get(idx) | unsafe_ptr()

# BitSet
bs = BitSet[size: Int=128]()
bs.set(idx: Int) | clear(idx: Int) | toggle(idx: Int) | test(idx: Int)
len(bs)
bs.union(other) | intersection(other) | difference(other)

# Optional
opt = Optional(value) | Optional[Int](None)
opt.value() | unsafe_value() | take() | unsafe_take() | or_else(default)
if opt: pass
for item in opt: print(item)

# Interval
interval = Interval(start, end)
interval.overlaps(other) | union(other) | intersection(other)
val in interval
```

### Comprehensions
```mojo
var nums = [1, 2, 3, 4]
var evens = [n for n in nums if n % 2 == 0]
```

### Slicing
```mojo
slice(end) | slice(start, end) | slice(start, end, step)
Slice(Optional[Int], Optional[Int], Optional[Int])
slice_obj.indices(length)

# ContiguousSlice / StridedSlice for specialization
ContiguousSlice(start, size)
StridedSlice(start, size, stride)

# List slicing without stride returns Span (no allocation)
span = list[1:5]
```

## 3. Control Flow

### Loops
```mojo
for i in range(3):
    print(i)
else:
    print("finished")  # runs if loop wasn't broken

# Note: match/switch is not supported yet
```

### Bool Operations
```mojo
all(iterable) | any(iterable)
all(simd_vector) | any(simd_vector)
all(map(fn, iterable)) | any(map(fn, iterable))
```

## 4. Pointers & Memory

### UnsafePointer

Dynamically allocate/free memory, interface with C/FFI, build data structures. Inherently unsafe - you manage allocation, initialization, and freeing.

**Lifecycle States:**
```
Uninitialized → Null (addr 0) → Allocated → Initialized → Dangling
```

```mojo
from memory import UnsafePointer, alloc

# Allocation
ptr = alloc[Int](count)                    # Allocate space for count values
ptr = alloc[Float32](256, alignment=64)    # With alignment
ptr = UnsafePointer[Int, MutOrigin.external]()  # Null pointer

# Initialization (allocated memory is uninitialized)
ptr.init_pointee_copy(value)               # Copy value into memory
ptr.init_pointee_move(value^)              # Move value into memory
ptr = UnsafePointer(to=existing_value)     # Point to existing value (no alloc needed)
ptr = UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=mmio_address)  # From raw address

# Dereferencing (memory must be initialized)
value = ptr[]                              # Read pointee
ptr[] = new_value                          # Write pointee
ptr[3] = value                             # Subscript access for arrays

# Destruction
ptr.destroy_pointee()                      # Requires implicitly-destructible pointee
value = ptr.take_pointee()                 # Move out, leave uninitialized
ptr.init_pointee_move_from(src_ptr)        # Move from src to self, src uninitialized
swap_pointees(ptr1, ptr2)
ptr.free()                                 # Deallocate (no destructors called!)

# Linear pointees: use destroy_pointee_with(dtor_fn_ptr)

# Pointer arithmetic
offset_ptr = ptr + 2
ptr += 1
ptr -= 1

# SIMD load/store
values = ptr.load[width=4]()               # Load SIMD vector
ptr.store(values)                          # Store SIMD vector
values = ptr.strided_load[width=8](stride) # Load with stride (e.g., RGB channels)
ptr.strided_store[width=8](values, stride)
values = ptr.gather(offsets)               # Gather from offset vector
ptr.scatter(values, offsets)               # Scatter to offset vector
ptr.store[volatile=True](value)            # Volatile store (for MMIO)

# Type casting
new_ptr = ptr.bitcast[NewType]()           # Same address, different type
safe_cast = ptr.as_any_origin() | as_immutable()
unsafe_cast = ptr.unsafe_mut_cast[True]() | unsafe_origin_cast[new_origin]()
```

**Origin Tracking:**
```mojo
# alloc() returns MutOrigin.external (untracked by lifetime checker)
# UnsafePointer(to=value) infers origin from value

fn unsafe_ptr(ref self) -> UnsafePointer[T, origin_of(self)]:
    return self.data.unsafe_origin_cast[origin_of(self)]()
```

**Foreign Interop:**
```mojo
# Python
ptr = arr.ctypes.data.unsafe_get_as_pointer[DType.int64]()

# C/C++ FFI
ptr = external_call["c_func", UnsafePointer[Int, MutOrigin.external]]()

# Opaque pointer (void* equivalent)
comptime OpaquePointer = UnsafePointer[NoneType]
opaque = ptr.bitcast[NoneType]()
```

**Byte Order:**
```mojo
swapped = byte_swap(value)                 # Little ↔ big endian
```

### Other Pointer Types
```mojo
ptr = OwnedPointer(value)                  # Single-owner heap allocation
value = ptr[]
shared = ArcPointer(value)                 # Reference-counted shared ownership
copy = shared
```

### Stack Allocation
```mojo
from memory import stack_allocation

var buf = stack_allocation[256, DType.int8]()
var aligned = stack_allocation[64, DType.float32, alignment=64]()
var typed = stack_allocation[count, MyType]()
# No free() required - deallocated when scope exits
```

### When to Use Pointers vs Tensors

**UnsafePointer** - Low-level escape hatch:
- C/FFI interop, manual memory management
- YOU manage: bounds, lifetime, shape
- Stdlib functions require wrapper patterns

**LayoutTensor** - Type-safe tensor with explicit layout:
- Multi-dimensional data with compile-time layout specification
- Works with stdlib via function-based APIs (sum, reduce, etc.)
- Supports CPU and GPU operations
- Layout separates logical structure from memory organization

# LayoutTensor Reference

**Multi-dimensional tensor view with compile-time layout and origin tracking. Does not own underlying memory.**

## Origin vs LayoutTensor

**Origin** - Compiler token for memory lifetime tracking:
- Prevents use-after-free, ensures safe aliasing
- Generic parameter in function signatures: `ref [origin] data`
- Common origins: `MutAnyOrigin`, `ImmutAnyOrigin`, `external`
- Use `origin_of(value)` in generic code when needed

**Layout** - Compile-time memory organization:
- Defines element arrangement: `Layout.row_major(M, K)`, `Layout.col_major(M, K)`
- Enables optimizer for vectorization/coalescing
- Tiled layouts for cache efficiency

**LayoutTensor** - Combines layout + origin + data pointer

## Construction

```mojo
from layout import Layout, LayoutTensor
from collections import InlineArray

# 2D row-major tensor
comptime M = 4
comptime K = 8
comptime layout = Layout.row_major(M, K)
var storage = InlineArray[Float32, M * K](fill=0.0)
var tensor = LayoutTensor[DType.float32, layout](storage)

# 1D as row vector (1, SIZE)
comptime vec_layout = Layout.row_major(1, 16)
var vec_storage = InlineArray[Float32, 16](fill=1.0)
var vec = LayoutTensor[DType.float32, vec_layout](vec_storage)

# Column-major layout
comptime col_layout = Layout.col_major(M, K)
var col_tensor = LayoutTensor[DType.float32, col_layout](storage)
```

## Element Access

```mojo
# Read (returns SIMD[dtype, 1], extract scalar with [0])
var element = tensor[2, 3][0]

# Write
tensor.store(2, 3, SIMD[DType.float32, 1](42.0))

# SIMD load/store
var vals = tensor.load[width=4](1, 0)  # Load 4 elements
tensor.store(1, 0, vals * 2.0)
```

## Stdlib Integration (Function-Based APIs)

```mojo
from algorithm import sum, vectorize

# sum() via parametric closure
comptime lt = Layout.row_major(1, 16)
var storage = InlineArray[Float32, 16](fill=3.0)
var tensor = LayoutTensor[DType.float32, lt](storage)

@parameter
fn input_fn[dtype_: DType, width: Int](idx: Int) -> SIMD[dtype_, width]:
    return tensor.load[width=width](0, idx).cast[dtype_]()

var result = sum[DType.float32, input_fn](16)  # Returns 48.0

# vectorize() pattern
var total = Float32(0)

@parameter
fn accumulate[width: Int](idx: Int):
    total += tensor.load[width=width](0, idx).reduce_add()

vectorize[accumulate, 4](16)
```

## Tiling & Iteration

```mojo
# Extract tile
comptime TILE_M = 4
comptime TILE_K = 4
var tile = tensor.tile[TILE_M, TILE_K](1, 1)  # Tile at position (1,1)

# Manual tile iteration
comptime num_tiles = 16 // 4
for tile_idx in range(num_tiles):
    var tile_offset = tile_idx * 4
    var tile_vals = tensor.load[width=4](0, tile_offset)
    # Process tile
```

## Vectorization & Distribution

```mojo
# Vectorize for SIMD-aligned access
comptime simd_width = 4
var v_tensor = tensor.vectorize[1, simd_width]()

# Distribute across threads (GPU pattern)
comptime thread_layout = Layout.row_major(8, 4)
var fragment = tensor.distribute[thread_layout](thread_id)
```

## Generic Functions with Origin

```mojo
# Accept any origin - caller's origin propagated
fn process[
    layout: Layout,
    origin: Origin
](tensor: LayoutTensor[DType.float32, layout, origin]):
    var val = tensor[0, 0][0]

# Generic function for stdlib
fn tensor_sum[
    layout: Layout,
    origin: Origin
](tensor: LayoutTensor[DType.float32, layout, origin]) -> Float32:
    @parameter
    fn input_fn[dtype_: DType, width: Int](idx: Int) -> SIMD[dtype_, width]:
        return tensor.load[width=width](0, idx).cast[dtype_]()

    return sum[DType.float32, input_fn](tensor.shape[1]())
```

## Key Patterns

```mojo
# Matmul pattern
for m in range(M):
    for n in range(N):
        var acc = Float32(0)
        for k in range(K):
            acc += A[m, k][0] * B[k, n][0]
        C.store(m, n, SIMD[DType.float32, 1](acc))

# Row/column access
var row = tensor.load[width=K](2, 0)  # Load row 2
var col_sum = Float32(0)
for i in range(M):
    col_sum += tensor[i, 3][0]  # Sum column 3
```

### Memory Operations
```mojo
parallel_memcpy[dtype](dest, src, count, count_per_task, num_tasks)
buf.zero() | fill(value) | tofile(path)
```

## 5. SIMD & Vectorization

### SIMD Operations
```mojo
vectorize[func, simd_width, unroll_factor=2, size=1024]()
vectorize[func, simd_width](size)

@parameter
fn closure[width: Int](i: Int):
    ptr.store[width=width](i, value)

bit_reverse(val)
count_leading_zeros(val), count_trailing_zeros(val)
pop_count(val)
rotate_bits_left[shift](x)

next_power_of_two(val), prev_power_of_two(val)
log2_ceil(val), log2_floor(val)
```

### SIMD Methods
```mojo
vec.shuffle[*mask: Int](other) - permute/blend elements
vec.interleave(other) | deinterleave(other) - zip/unzip elements
vec.join(other) - concatenate two vectors
vec.slice[offset, size]() - extract subvector
vec.insert(other, offset) - insert elements
vec.rotate_left[shift]() | rotate_right[shift]() - element rotation
vec.reduce_add() | reduce_mul() | reduce_max() | reduce_min() - reductions
vec.cast[target_type]() - type conversion
vec.fma(y, z) - fused multiply-add (x * y + z)
mask.select(true_val, false_val) - conditional select
iota[dtype, width]() - sequential values [0, 1, 2, ...]
any(bool_vec) | all(bool_vec) - boolean reductions
vec.eq(other) | ne | lt | le | gt | ge - comparisons
```

### Parallelization
```mojo
parallelize[func](num_work_items, num_workers)
elementwise[func, simd_width, target="gpu"](shape, device_context)
tile[workgroup_fn, tile_sizes](offset, upperbound)
tile[workgroup_fn, sizes_x, sizes_y](off_x, off_y, bound_x, bound_y)
```

## 6. Low-Level GPU Programming

### Imports
```mojo
from gpu import block_idx, thread_idx, cluster_sync
from gpu.primitives.warp import shuffle_idx, shuffle_up, shuffle_down, shuffle_xor
from gpu.primitives.block import sum, max, min, broadcast, prefix_sum
from gpu.compute.mma import mma, load_matrix_a, load_matrix_b, store_matrix_d
from gpu.memory import async_copy, async_copy_commit_group, async_copy_wait_group
from gpu.sync import barrier, syncwarp, named_barrier
from gpu.sync.semaphore import Semaphore
```

### Thread Hierarchy
```mojo
thread_idx.x|y|z, block_idx.x|y|z, block_dim.x|y|z, grid_dim.x|y|z
global_idx.x|y|z, cluster_idx.x|y|z, cluster_dim.x|y|z
block_id_in_cluster.x|y|z, block_rank_in_cluster()
lane_id(), warp_id(), sm_id()
```

### Memory Operations
```mojo
# AddressSpace.GENERIC | GLOBAL | SHARED | CONSTANT | LOCAL | SHARED_CLUSTER

val = load[
    dtype, width=1, read_only=False, prefetch_size=None,
    cache_policy=ALWAYS|GLOBAL|STREAMING|VOLATILE|LAST_USE|
                 WRITE_BACK|WRITE_THROUGH|WORKGROUP,
    eviction_policy=EVICT_FIRST|EVICT_LAST|EVICT_NORMAL|NO_ALLOCATE,
    alignment
](ptr)

val = load_relaxed[dtype](ptr, scope=THREAD|WARP|BLOCK|CLUSTER|GPU|SYSTEM)
val = load_acquire[dtype](ptr, scope)
val = load_volatile[dtype](ptr)

store_relaxed[dtype](ptr, value, scope)
store_release[dtype](ptr, value, scope)
store_volatile[dtype](ptr, value)
```

### Async Copy
```mojo
async_copy[
    dtype, size, fill=None, bypass_L1_16B=True,
    l2_prefetch=None, eviction_policy
](src_global, dst_shared, src_size, predicate=False)

async_copy_commit_group()
async_copy_wait_group(n)
async_copy_wait_all()
```

### Synchronization
```mojo
barrier()
syncwarp(mask=-1)
named_barrier[num_threads](id=0)
named_barrier_arrive[num_threads](id=0)

mbarrier_init[type](shared_mem, num_threads)
state = mbarrier_arrive[type](shared_mem)
mbarrier_arrive_expect_tx_shared[type](addr, tx_count)
state = mbarrier_arrive_expect_tx_relaxed[type, scope=BLOCK, space=BLOCK](
    addr, tx_count
)
ok = mbarrier_test_wait[type](shared_mem, state)
mbarrier_try_wait_parity_shared[type](addr, phase, ticks)
async_copy_arrive[type, address_space](address)

cluster_arrive() | cluster_arrive_relaxed()
cluster_wait() | cluster_sync() | cluster_sync_relaxed()
cluster_sync_acquire() | cluster_sync_release()

# AMD-specific
s_waitcnt[vmcnt, expcnt, lgkmcnt]()
s_waitcnt_barrier[...]()
schedule_barrier(mask=NONE|ALL_ALU|VALU|SALU|MFMA|ALL_VMEM|VMEM_READ|VMEM_WRITE|ALL_DS|DS_READ|DS_WRITE|TRANS)
schedule_group_barrier(mask, size, sync_id)

launch_dependent_grids()
wait_on_dependent_grids()

threadfence[scope=GPU]()
```

### Warp Operations
```mojo
val = shuffle_idx[dtype, simd_width](val, offset)
val = shuffle_idx[dtype, simd_width](mask, val, offset)
val = shuffle_up/down/xor[dtype, simd_width](val, offset)

scalar = sum(val) | max(val) | min(val)
val = broadcast[dtype, width](val)

val = lane_group_sum/max/min[dtype, width, num_lanes, stride=1](val)
val = lane_group_sum_and_broadcast[dtype, width, num_lanes, stride=1](val)
val = lane_group_max_and_broadcast[dtype, width, num_lanes, stride=1](val)
val = lane_group_reduce[dtype, width, shuffle, func, num_lanes, stride=1](val)
val = prefix_sum[dtype, intermediate_type=dtype, output_type=dtype, exclusive=False](x)

mask = vote[ret_type](val)
```

### Block Operations
```mojo
val = sum/max/min[dtype, width, block_size, broadcast=True](val)
val = broadcast[dtype, width, block_size](val, src_thread=0)
val = prefix_sum[dtype, block_size, exclusive=False](val)
```

### Tensor Core MMA
```mojo
mma[block_size=1](mut d: SIMD, a: SIMD, b: SIMD, c: SIMD)

load_matrix_a[m=16, n=8, k=8](ptr, tile_row, tile_col, ldm)
load_matrix_b[m=16, n=8, k=8](ptr, tile_row, tile_col, ldm)
store_matrix_d[dtype, m, n, k](ptr, d: SIMD[dtype,4], tile_row, tile_col, ldm)

ld_matrix[dtype, simd_width, transpose=False](ptr)
st_matrix[dtype, simd_width, transpose=False](ptr, d)
```

### AMD Buffer
```mojo
buf = AMDBufferResource.__init__[dtype](gds_ptr, num_records)
val = buf.load[dtype, width, cache_policy](vector_offset, scalar_offset=0)
buf.store[dtype, width, cache_policy](vector_offset, val, scalar_offset=0)
buf.load_to_lds[dtype, width, cache_policy](vector_offset, shared_ptr, scalar_offset=0)
```

### CPU Intrinsics
```mojo
init_intel_amx()
tile = __tile[rows, cols, dtype]()

result = dot_i8_to_i32_x86[width](src, a, b)
result = dot_i8_to_i32_saturated_x86[width](src, a, b)
result = dot_i8_to_i32_AVX2[width](src, a, b)
result = dot_i16_to_i32_x86[width](src, a, b)
result = dot_i16_to_i32_AVX2[width](src, a, b)

fma16/fma32/fma64/mac16(gpr)
ldx/ldy/ldz/stx/sty/stz/extrx/extry(gpr)
```

### GPU Intrinsics
```mojo
val = byte_permute(a, b, c) | lop[lut](a, b, c) | mulhi(a, b) | mulwide(a, b)
val = permlane_shuffle[dtype, simd_width, stride](val)
val = permlane_swap[dtype, stride](val1, val2)
val = ds_read_tr16_b64[dtype](shared_ptr)
warpgroup_reg_alloc[count]() | warpgroup_reg_dealloc[count]()
```

## 7. Layout Programming

### Layout
```mojo
from layout import Layout
from layout.layout_tensor import LayoutTensor, LayoutTensorIter, ThreadScope
from layout.runtime_layout import RuntimeLayout, make_layout, coalesce
from layout.runtime_tuple import RuntimeTuple
from layout.swizzle import Swizzle, ComposedLayout, make_swizzle

# Mojo Layout
layout = Layout.row_major(rows, cols) | col_major(rows, cols)
layout = Layout(shape_tuple, stride_tuple)
layout = tile_to_shape(tile_layout, final_shape)
layout = blocked_product(tile, tiler)
layout = make_ordered_layout(shape, order)

idx = layout(coords)
coords = layout.idx2crd(idx)
size = layout.size(), cosize = layout.cosize(), rank = layout.rank()

# Runtime Layout
layout = RuntimeLayout[layout, element_type, linear_idx_type]()
layout = RuntimeLayout[layout, ...](shape, stride)
layout = row_major/col_major[rank](shape)

idx = layout(i) | layout[t](idx_tuple)
coords = layout.idx2crd[t](idx)
size = layout.size(), dim = layout.dim(i)
casted = layout.cast[dtype]()
sub = layout.sublayout[i]()
coalesced = coalesce[l, keep_rank](layout)
combined = make_layout[l1, l2](a, b)

# Swizzle
swizzle = Swizzle(bits, base, shift)
result = swizzle(index | offset)
swizzle = make_swizzle[num_rows, row_size, access_size]()
swizzle = make_swizzle[dtype, mode]()
swizzle = make_ldmatrix_swizzle[dtype, row_size, log2_vector_width]()

composed = ComposedLayout[LayoutA, LayoutB, offset](layout_a, layout_b)
result = composed(idx, offset_val)
```

### LayoutTensor
```mojo
iter = LayoutTensorIter[mut, dtype, layout, origin](
    ptr, bound, stride|runtime_layout, offset=0
)
tensor = iter[self] | get()
next_iter = iter.next(steps) | next_unsafe(steps)
iter += steps
reshaped = iter.reshape[dst_layout]()
casted = iter.bitcast[new_type]()

# ThreadScope
ThreadScope.BLOCK | WARP

# Memory Copy Operations
copy_dram_to_local[
    src_thread_layout, num_threads, thread_scope,
    block_dim_count, cache_policy
](dst, src, src_base, offset|bounds)

copy_dram_to_sram[
    src_thread_layout, dst_thread_layout, swizzle,
    num_threads, thread_scope, block_dim_count
](dst, src)

copy_dram_to_sram_async[
    src_thread_layout, dst_thread_layout, swizzle,
    fill, eviction_policy, num_threads, block_dim_count
](dst, src)

copy_local_to_dram[
    dst_thread_layout, num_threads, thread_scope, block_dim_count
](dst, src|dst_base)

copy_local_to_local(dst, src)

copy_local_to_shared[
    thread_layout, swizzle, num_threads,
    thread_scope, block_dim_count, row_major
](dst, src)

copy_sram_to_dram[
    thread_layout, swizzle, num_threads, block_dim_count, binary_op
](dst, src)

copy_sram_to_local[src_warp_layout, axis](dst, src)

cp_async_k_major[dtype, eviction_policy](dst, src)
cp_async_mn_major[dtype, eviction_policy](dst, src)

# Math Operations
result = sum/max[axis](inp, outp)
result = sum/max[axis](inp)
result = max[dtype, layout](x, y)
scalar = mean(src)
mean[reduce_axis](src, dst)
scalar = variance(src, correction=1)
outer_product_acc(res, lhs, rhs)

# Mojo LayoutTensor
storage = InlineArray[Float32, size](uninitialized=True)
tensor = LayoutTensor[DType.f32, layout](storage)

dev_buf = ctx.enqueue_create_buffer[dtype](size)
tensor = LayoutTensor[dtype, layout](dev_buf)

tile = LayoutTensor[
    dtype, layout, MutAnyOrigin,
    address_space=AddressSpace.SHARED
].stack_allocation()

element = tensor[x, y][0]
elements = tensor.load[width](x, y)
tensor.store(x, y, values)

tile = tensor.tile[tile_h, tile_w](tile_row, tile_col)
iter = tensor.tiled_iterator[tile_h, tile_w, axis=1](row, col)
tile = iter[]
iter += 1

v_tensor = tensor.vectorize[1, simd_width]()
fragment = tensor.distribute[thread_layout](thread_id)
dst.copy_from(src)
shared.copy_from_async(global)
async_copy_wait_all()
```

### Tensor Core
```mojo
from layout.tensor_core import TensorCore, TiledTensorCore
from layout.tensor_core import get_mma_shape, get_fragment_size

tc = TensorCore[out_type, in_type, shape, transpose_b]()
a_frag = tc.load_a[swizzle](a|warp_tile, fragments, mma_tile_coord_k)
b_frag = tc.load_b[swizzle](
    b|warp_tile, fragments,
    mma_tile_coord_k|scales, warp_tile_coord_n|mma_tile_coord_k
)
c_frag = tc.load_c(c)
d = tc.mma_op(a, b, c) | tc.mma(a_frag, b_frag, c_frag)
tc.store_d(d_dst, d_src)

shapes = TensorCore.get_shapes[out_type, in_type]()
shape = get_mma_shape[input_type, accum_type, shape_id]()
frag_size = get_fragment_size[mma_shape]()

TiledTensorCore[
    out_type, in_type, shape, group_size, transpose_b
].mma[swap_a_b](a, b, c)
```

### Pipeline State
```mojo
from layout.tma_async import PipelineState, SharedMemBarrier

state = PipelineState[num_stages]()
state = PipelineState[num_stages](index, phase, count)
idx = state.index(), phase = state.phase()
state.step()
next_state = state.next()

barrier = SharedMemBarrier()
barrier.init(num_threads)
barrier.expect_bytes(bytes)
barrier.arrive_and_expect_bytes(bytes, cta_id, pred)
barrier.wait(phase) | wait_acquire[scope](phase)
barrier.arrive() | arrive_cluster(cta_id, count)
```

### RuntimeTuple
```mojo
t = RuntimeTuple[S, element_type]()
t = RuntimeTuple[S, element_type](*values | index_list)

elem = t[i], t[i] = val
scalar = t.get_int(), i = int(t)

concatenated = t.concat[R](rhs)
flat = t.flatten()
casted = t.cast[dtype]()

prod = product[t](tuple)
prefix = prefix_product[t](tuple)

idx = crd2idx[crd_t, shape_t, stride_t, out_type](crd, shape, stride)
coords = idx2crd[idx_t, shape_t, stride_t](idx, shape, stride|shape)
result = shape_div[a_t, b_t](a, b)

tuple = IntTuple(1, 2, 3) | IntTuple(IntTuple(2, 2), IntTuple(3, 3))
```

### DimList
```mojo
dims = DimList(2, 4, 8)
dims = DimList(Dim(), Dim(4), Dim())
dims = DimList.create_unknown[rank]()

dims.product[length]() | product[start, end]()
dims.all_known[length]() | contains[length](value)
dims.into_index_list[rank]()

d = Dim(x) | Dim(x, y) | Dim(x, y, z) | Dim((x, y, z))
d.x() | y() | z()
d[0] | [1] | [2]
```

## 8. Reductions

```mojo
reduce[map_fn, reduce_fn, reduce_axis](src, dst, init)
reduce_boolean[reduce_fn, continue_fn](src, init)
sum[dtype, input_fn, output_fn](input_shape, reduce_dim, context)
mean[dtype, input_fn, output_fn](input_shape, reduce_dim, output_shape, context)
```

## 9. File System & OS

### Path Operations
```mojo
from os.path import (
    join, basename, dirname, split, splitroot, split_extension,
    exists, isdir, isfile, islink, lexists, is_absolute,
    expanduser, expandvars, realpath, getsize
)

path = join("/a", "b", "c")
name = basename("/path/file.txt")
dir = dirname("/path/file.txt")
head, tail = split("/path/file.txt")
drive, root, tail = splitroot("C:/path/file")
root, ext = split_extension("file.tar.gz")

home = expanduser("~/docs")
exp = expandvars("$HOME/${USER}")
real = realpath("/path/../symlink")
size = getsize("/path/file")

exists(path), isdir(path), isfile(path), islink(path)
lexists(path), is_absolute(path)
```

### File Operations
```mojo
from os import (
    listdir, mkdir, makedirs, rmdir, removedirs, remove, unlink,
    stat, lstat, getuid, isatty, link, symlink
)

entries = listdir("/dir")
mkdir("/dir", mode=0o755)
makedirs("/deep/nested/dir", mode=0o755, exist_ok=True)
rmdir("/empty/dir")
removedirs("/empty/parent/child")
remove("/file")
unlink("/file")
link("/source", "/dest")
symlink("/target", "/linkpath")

info = stat("/path")  # Follow symlinks
linfo = lstat("/symlink")  # Don't follow

struct stat_result:
    st_mode: Int, st_ino: Int, st_dev: Int, st_nlink: Int
    st_uid: Int, st_gid: Int, st_size: Int
    st_atimespec: _CTimeSpec, st_mtimespec: _CTimeSpec
    st_ctimespec: _CTimeSpec, st_birthtimespec: _CTimeSpec
    st_blocks: Int, st_blksize: Int, st_rdev: Int, st_flags: Int

uid = getuid()  # Linux/macOS
is_tty = isatty(fd: Int)

# File type checks
from stat import S_ISREG, S_ISDIR, S_ISCHR, S_ISBLK, S_ISFIFO, S_ISLNK, S_ISSOCK
from stat import S_IFMT, S_IFREG, S_IFDIR, S_IFCHR, S_IFBLK, S_IFIFO, S_IFLNK, S_IFSOCK

S_ISREG(info.st_mode)  # Regular file
S_ISDIR(info.st_mode)  # Directory
```

### Path Object
```mojo
from pathlib import Path, cwd

p = Path() | Path("/abs") | Path("rel")
sub = p / "dir" / "file"
p /= "dir"
if p == Path("/same") or p == "/same": pass

p.exists(), p.is_dir(), p.is_file()
s = p.stat(), p.lstat()
txt = p.read_text()
bytes = p.read_bytes()
p.write_text("text"), p.write_bytes(span)

name = p.name()  # basename
ext = p.suffix()  # extension
parts = p.parts()  # List[StringSlice]
joined = p.joinpath("a", "b", "c")
dirs = p.listdir()  # List[Path]
expanded = p.expanduser()
home = Path.home()
pwd = cwd()
```

### Environment Variables
```mojo
from os import getenv, setenv, unsetenv

val = getenv("VAR", default="")
ok = setenv("VAR", "value", overwrite=True)
ok = unsetenv("VAR")
```

### User Info (Linux/macOS)
```mojo
from pwd import getpwnam, getpwuid, Passwd

struct Passwd:
    pw_name: String, pw_passwd: String
    pw_uid: Int, pw_gid: Int
    pw_gecos: String, pw_dir: String, pw_shell: String

user = getpwnam("user")
user = getpwuid(uid)
```

### OS Constants
```mojo
os.sep = "/"
os.SEEK_SET = 0, os.SEEK_CUR = 1, os.SEEK_END = 2
```

## 10. I/O

```mojo
fh = FileHandle(path, mode="r"|"w"|"rw"|"a")
data = fh.read(size=-1)
data = fh.read[dtype, origin](buffer)
data = fh.read_bytes(size=-1)
pos = fh.seek(offset, whence=0)
fh.write_bytes(bytes) | write[*Ts](*args) | close()

fd = FileDescriptor(value=1)
fd.write_bytes(bytes) | read_bytes(buffer) | write[*Ts](*args)
ok = fd.isatty()

fh = open[PathLike](path, mode)
s = input(prompt="")
print[*Ts](*values, sep=" ", end="\n", flush=False, file=FileDescriptor(1))
```

## 11. Iterators

```mojo
# Built-ins / prelude
enumerate(ref iterable, start=0)
zip(ref iterable_a, ref iterable_b)

# Iterator adapters
map[IterableType, ResultType, function](ref iterable)

# Extras
from iter import peekable
from itertools import count, repeat, product
```

```mojo
from iter import Iterator, StopIteration

# Iterator protocol:
# - implement __next__ that raises StopIteration
# - __has_next__ was removed
# - if you want `for x in MyIter()`, implement a consuming __iter__
struct MyIter(Iterator):
    comptime Element = Int
    var i: Int

    fn __init__(out self):
        self.i = 0

    fn __iter__(var self) -> Self:
        return self^

    fn __next__(mut self) raises StopIteration -> Int:
        if self.i >= 3:
            raise StopIteration()
        self.i += 1
        return self.i
```

## 12. Logging

```mojo
logger = Logger[
    level=Level.NOTSET|TRACE|DEBUG|INFO|WARNING|ERROR|CRITICAL
](fd, prefix="", source_location=False)

logger.trace|debug|info|warning|error|critical[*Ts](
    *values, sep=" ", end="\n"
)
```

## 13. Math

```mojo
val = ceil[T](value) | floor[T](value) | trunc[T](value)
val = ceildiv[T](numerator, denominator)
val = align_up|align_down(value, alignment)
val = clamp(val, lower_bound, upper_bound)
val = copysign[dtype, width](magnitude, sign)
val = acos|asin|atan[dtype, width](x)
val = atan2[dtype, width](y, x)
val = acosh|asinh|atanh[dtype, width](x)
val = cbrt[dtype, width](x)

# Constants
pi=3.14159265, tau=6.28318531, e=2.71828182, log2e=1.44269504
```

## 14. Random

```mojo
from random import Random, NormalRandom  # Philox-based RNG

rng = Random[rounds=10](seed=0, subsequence=0, offset=0)
val = rng.step() | step_uniform()

nrng = NormalRandom[rounds=10](seed=0, subsequence=0, offset=0)
val = nrng.step_normal(mean=0, stddev=1)
```

## 15. Benchmarking

```mojo
var bench = Bench(BenchConfig(max_iters=100))
bench.bench_function[func](BenchId("name"), measures)
bench.bench_with_input[T, func](BenchId("name"), input, measures)
print(bench)
bench.dump_report()

ThroughputMeasure(BenchMetric.elements|bytes|flops, count)

bencher = Bencher(num_iters)
bencher.iter[func]() | iter_custom[func]() | iter_custom[kernel_fn](device_context)

BenchConfig(
    min_runtime_secs=0.1, max_runtime_secs=1.0,
    num_warmup_iters=10, max_batch_size=0,
    flush_denormals=True
)
```

## 16. Assertions & Debug

```mojo
constrained[cond, msg]()
debug_assert(cond, msg)
debug_assert[assert_mode="safe"](cond, msg)
debug_assert[check_fn, cpu_only=True](msg)
debug_assert[_use_compiler_assume=True](cond, msg)
```

## 17. Optimization

```mojo
@register_passable(trivial)
struct Point:
    var x: Float32
    var y: Float32

alignment2 = simd_width * sizeof(dtype)
buf.prefetch[PrefetchOptions(locality=3, cache=.data, rw=.read)](idx)
keep(value)
clobber_memory()
```

## 18. Utilities

```mojo
ptr = external_memory[dtype, address_space, alignment, name]()
with ProfileBlock[enabled=False]("name"): pass

from gpu.sync.semaphore import Semaphore
sem = Semaphore(lock, thread_id)
sem.fetch() | state() | wait(status=0) | release(status=0)

nbs = NamedBarrierSemaphore[thread_count, id_offset, max_num_barriers]
nbs.wait_eq(id, status=0) | wait_lt(id, count=0) | arrive_set(id, status=0)

# Hash
hash[T, HasherType=AHasher[0]](hashable)
hash[HasherType](bytes, n)

# Abort
abort[result: AnyType=None]() -> result
abort[result: AnyType=None, *, prefix: StringSlice="ABORT:"](message: String) -> result
```

## 19. Base64

```mojo
b64encode(input_bytes) | b64encode(input_string)
b64decode[validate=False](str)
b16encode(str), b16decode(str)
```

## 20. Variadics

```mojo
fn sum(*args: Int) -> Int:
    for i in range(len(args)):
        total += args[i]

fn sum[*T: Intable](*args: *T) -> Int:
    @parameter
    for i in range(args.__len__()):
        total += Int(args[i])
```

## 21. Compilation

```mojo
from compile import compile_info, get_linkage_name, get_type_name

info = compile_info[func, emission_kind="asm"|"llvm"|"llvm-opt"|"object"]()
get_linkage_name[func]()
get_type_name[type]()
```

### Reflection

```mojo
from compile import get_type_name
from reflection import (
    struct_field_count,
    struct_field_names,
    struct_field_types,
    struct_field_index_by_name,
    struct_field_type_by_name,
)

@fieldwise_init
struct Point(Copyable):
    var x: Float32
    var y: Float32

fn print_fields[T: AnyType]():
    comptime names = struct_field_names[T]()
    comptime types = struct_field_types[T]()
    @parameter
    for i in range(struct_field_count[T]()):
        print(names[i], get_type_name[types[i]]())

fn main():
    print_fields[Point]()
    comptime idx = struct_field_index_by_name[Point, "x"]()
    comptime field_type = struct_field_type_by_name[Point, "y"]()
    print(idx)
    print(get_type_name[field_type.T]())
```

## 22. Coroutines

### Mojo Coroutines & Async

### Coroutine Struct (Built-in)
```mojo
# Coroutine[type, origins] - NOT copyable, use ^ to move
async fn compute() -> Int: return 42

async fn caller():
    var coro = compute()
    var result = await coro^  # Move with ^
```


### Runtime AsyncRT
```mojo
from runtime.asyncrt import Task, TaskGroup, create_task, parallelism_level

# Task - Scheduled async execution
var task: Task[Int, {}]
create_task(work(), task)
var result = await task^

# TaskGroup - Parallel execution
var group = TaskGroup()
group.create_task(async_fn1())
group.create_task(async_fn2())
await group  # or group.wait() for blocking

var cores = parallelism_level()
```

### Methods
```mojo
# Coroutine[type, origins]
__init__(handle: !co.routine)
__await__(var self, out result: type)
force_destroy(var self)

# RaisingCoroutine[type, lifetimes] - For raises functions
async fn work() raises -> Int: ...
```

### Entry Point
```mojo
fn main():
    async_main()()  # Call async fn + execute coroutine
```

## 23. Module System

```mojo
import module
from module import Item
from package.module import Item
import module as alias
```
## 24. Pixi Package Manager

### Installation
```sh
curl -fsSL https://pixi.sh/install.sh | bash
pixi self-update
```

### Configuration
```sh
# Install Mojo & MAX toolchain (Pixi)
curl -fsSL https://pixi.sh/install.sh | sh
pixi init myproj -c https://conda.modular.com/max-nightly/ -c conda-forge && cd myproj
pixi add mojo
pixi run mojo --version
```
### Package Management
```sh
pixi add mojo                # latest
```

### Execution
```sh
pixi run mojo --version
pixi shell  # interactive shell
exit
```
