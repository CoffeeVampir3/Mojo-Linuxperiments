Many languages have facilities for metaprogramming: writing code that generates or modifies code. Python has facilities for dynamic metaprogramming: features like decorators, metaclasses, and many more. These features make Python very flexible and productive, but since they're dynamic, they come with run-time overhead. Other languages have static or compile-time metaprogramming features, like C preprocessor macros and C++ templates. These can be limiting and hard to use.

Mojo's compile-time metaprogramming system uses the same language as run-time programs, so you don't have to learn a new language—just a few new features. The primary features you'll need to learn are:

    Compile-time statements and expressions
    Compile-time parameters
    Generics and traits

Compile-time statements and expressions

The comptime keyword identifies a statement or expression that needs to be evaluated at compile time. For example, the comptime keyword is used to declare compile-time constant values and to introduce compile-time conditionals and loops. For information on compile-time assignments and control flow, see Compile-time evaluation.
Compile-time parameters

Functions and structs can be parameterized with compile-time parameters, allowing you to define a container that holds different data types, or a matrix multiplication algorithm that's parameterized by the matrix dimensions. Compile-time parameters are similar to C++ template parameters or Rust generic parameters. At compile time, Mojo specializes parameterized code to make concrete versions—that is, it replaces parameters with constant values.

For example, a matrix multiplication function parameterized on its matrix dimensions can be specialized at compile time to select the most efficient algorithm based on those dimensions. For information on parameterization, see Parameters.
Traits and generics

Generic functions and structs are made to handle different data types—for example, a list that can hold Int, Float32, or String values. In Mojo, you build generics using compile-time parameters—generic functions and structs are parameterized on types.

A trait defines a set of shared behaviors for structs. Generic code uses traits to identify the behaviors it requires. For example, instead of being bound to a specific type, a sort function could require that the type being sorted is Movable and Comparable. For more information, see the sections on traits and generics.


Compile-time evaluation

To understand Mojo's metaprogramming, you need to understand how Mojo runs code at compile time. Several things can trigger compile-time code execution:

    Assigning an expression to a comptime value.
    Evaluating a comptime conditional or loop.
    Assigning an expression to a compile-time parameter.
    And a few less common cases, all identified with the comptime keyword.

Here are some examples:

comptime SIZE = 1024 // 32

Here the expression 1024 // 32 invokes the IntLiteral.__floordiv__() method. Since it occurs in a comptime assignment, the method must be run at compile time.

comptime for i in range(4):
   print(i)

Here the range(4) function needs to run to produce an iterator for the comptime for statement.

var array = InlineArray[Int, get_array_size()]()

In this example, the get_array_size() function needs to run at compile time to determine the size parameter, which forms part of the type of array. (For example, if get_array_size() returns 32, the type of the array variable is InlineArray[Int, 32].)

When the compiler encounters a function call in a compile-time context, the compiler runs the function separately, as if it was a small separate program. This is similar in concept to how C++ evaluates a constexpr. (For a slightly deeper look at this process, see How the compiler runs code.)

While most code can run at compile time, Mojo won't run code that depends on the execution environment. The following are examples of code that Mojo won't run at compile time:

    File I/O.
    Foreign function calls (for example, to external libraries).
    Functions that can raise errors.

In addition, the compiler can't run functions on the GPU. Compile-time functions in GPU code are actually run on the CPU.

When running code, the compiler can allocate memory and instantiate types that allocate memory, such as strings and collections. With some limitations, it can pass compile-time values on to run-time code, a process called materialization. For more information, see the section on materialization.
comptime values

It is very common to want to name compile-time values. Whereas var defines a runtime value, we need a way to define a named compile-time constant. For this, Mojo uses a comptime declaration. At its simplest, comptime can be used to define a constant value:

comptime rows = 512

A comptime value is always evaluated at compile time, so you can use comptime to force a function to run at compile time. You can use this to calculate constant values based on information available at compile time, such as hardware parameters.

comptime block_size = _calculate_block_size()

Types are another common use for comptime values. Because types are compile-time expressions, you can use a comptime value as a shorthand name for a parameterized type:

comptime Float16 = SIMD[DType.float16, 1]
comptime UInt8 = SIMD[DType.uint8, 1]

var x: Float16 = 0  # Float16 works like a named shorthand

(These shorthands and others are actually defined in the simd module.)

You can also parameterize a comptime value to express more complicated relationships. For details, see Parameterized comptime values.
Compile-time scope

Like var variables, comptime values obey scope, and you can use local comptime values within functions as you'd expect. Unlike var variables, comptime values can be defined at the module level, outside of any function.

The following constructs create a new compile-time scope:

    Functions. The body of a function creates a new compile-time scope.
    Compile-time flow control. Each branch of a compile-time conditional creates its own scope. The body of a comptime for loop also creates its own scope.

You can only assign a comptime value to a given identifier once in a given scope.

comptime VALUE = 10


def scope_me():
    print(VALUE)  # prints 10
    comptime VALUE = 20
    # comptime VALUE = 30  # error: invalid redeclaration of VALUE
    comptime if True:
        comptime VALUE = 40
        print(VALUE)  # prints 40
    print(VALUE)  # prints 20

Compile-time flow control

One of the simplest things you can do with metaprogramming is using compile-time flow control to conditionalize or repeat code. Some sample uses include:

    Conditionalizing platform-specific code (CPU vs. GPU, Linux vs. macOS) without runtime overhead.
    Unrolling loops to eliminate runtime branches.
    Handling different data types in generic code.

Unlike run-time flow control constructs, compile-time flow control constructs are evaluated once, at compile time, and determine what code is actually compiled.
Compile-time conditionals

You can add the comptime keyword to any if condition that's based on a valid compile-time expression (an expression that can be evaluated at compile time). This ensures that only the live branch of the if statement is compiled into the program, which can reduce your final binary size. For example:

from std.sys import has_accelerator

def main() raises:
    comptime if has_accelerator():
        run_on_gpu()
    else:
        run_on_cpu()

In this example, if no accelerator is available, the run_on_gpu() function is never called, or even compiled.

The comptime if statement can include elif and else branches just like a standard if statement.
Compile-time loop unrolling

You can add the comptime keyword to a for loop to create a loop that's fully unrolled at compile time. You should generally use this only for loops with small loop bodies and low iteration counts.

The loop sequence must be a valid compile-time expression (that is, an expression that can be evaluated at compile time). For example, if you use for i in range(LIMIT), the expression range(LIMIT) defines the loop sequence. This is a valid compile-time expression if LIMIT is a parameter, comptime value, or integer literal.

The compiler fully unrolls the loop by replacing the for loop with LIMIT copies of the loop body. The induction variable is replaced with a compile-time constant value for each "iteration." For example:

comptime for i in range(1, 5):
    b[i-1] = a[i] + a[i-1]

This is effectively unrolled to the following run-time code:

b[0] = a[1] + a[0]
b[1] = a[2] + a[1]
b[2] = a[3] + a[2]
b[3] = a[4] + a[3]

This unrolled loop compiles to branchless machine code, unlike a normal for loop, which includes a bounds test at every iteration. This can be especially important on GPU, to avoid thread divergence.

The comptime for construct unrolls at the beginning of compilation, which can greatly expand both the code size and the compilation time.
How the compiler runs code

The process of evaluating compile-time code involves three components of the compiler:

    Parser. Parses the code into an intermediate representation (IR) and performs type checking.
    Interpreter. Runs code at compile time.
    Elaborator. Substitutes concrete values for compile-time parameters and produces concrete versions of parameterized functions and structs.

When the parser turns code into IR, it also replaces some very simple comptime expressions with their values, a process called constant folding. For example, the compiler can constant fold the expression 2 + 3 to 5. Standard library functions that are marked @always_inline("builtin") are constant foldable. Compile-time expressions that can't be constant folded persist and are evaluated in the elaborator.

When the elaborator encounters a function call in a compile-time context, it invokes the interpreter to run the function. The interpreter then checks whether the function being called has already been elaborated to produce a concrete, executable function. If not, the interpreter adds that function to the elaborator's work queue, and waits until it's done. Finally, the interpreter runs the concrete function—almost like it was a small separate program—and passes the return value back to the elaborator, which integrates it into the parsed IR.

When reading code, it's important to remember that when a function is being interpreted at compile time, the function has been concretized: compile-time conditionals have been processed, and compile-time constraints and assertions have been tested. This sometimes appears to contradict the expectation that your code runs in the order it appears in the function. For example, if your function includes a compile-time assertion that fails, compilation fails before the interpreter enters the function, so no part of the function is evaluated—even code that occurs before the assertion.

Parameterization

Many programming languages offer systems for writing generic or polymorphic code, which let you write code once, and generate efficient, specialized code at compile time.

Mojo's compile-time parameter system lets you define reusable code. A parameter is a compile-time input to a struct or function. Parameters appear in square brackets after the struct or function name. Parameters can take ordinary values, like Int or String:

def multiplier[factor: Int](x: Int):
    return x * factor

def main():
    comptime times_ten = multiplier[10]
    x10 = times_ten(3)

Parameters can also take types as values, so you can write generic code that works for multiple data types:

struct MyList[T: AnyType]:
    # ... implementation omitted

def main():
    var l = MyList[Int]()

Mojo's parameters are similar to C++ template parameters or Rust generic parameters.

In Mojo, "parameter" and "parameter expression" refer to compile-time values, and "argument" and "expression" refer to dynamic values—which can be evaluated either at compile time or at run time. This usage of "parameter" is probably different from what you're used to from other languages, where "parameter" and "argument" are often used interchangeably.

In addition to parameterizing structs and functions, you can also define parameterized comptime values.
Parameterized functions

To define a parameterized function, add parameters in square brackets ahead of the argument list. Each parameter is formatted just like an argument: a parameter name, followed by a colon and a type. In the following example, the function has a single parameter, count of type Int.

def repeat[count: Int](msg: String):
    comptime for i in range(count):
        print(msg)

The comptime keyword shown here causes the for loop to be fully unrolled at compile time. The comptime for requires the loop limits to be known at compile time. Since count is a parameter, range(count) can be calculated at compile time.

Calling a parameterized function, you provide values for the parameters, just like function arguments:

repeat[3]("Hello")

Hello
Hello
Hello

The compiler resolves the parameter values during compilation, and creates a concrete version of the repeat[]() function for each unique parameter value. After resolving the parameter values and unrolling the loop, the repeat[3]() function would be roughly equivalent to this:

def repeat_3(msg: String):
    print(msg)
    print(msg)
    print(msg)

This doesn't represent actual code generated by the compiler. By the time parameters are resolved, Mojo code has already been transformed to an intermediate representation in MLIR.

If the compiler can't resolve all parameter values to constant values, compilation fails.
Overloading on parameters

Functions and methods can be overloaded on their parameter signatures. For information on overload resolution, see Overloaded functions.
Parameters at a glance

Parameters to a function or struct appear in square brackets after a function or struct name. Parameters always require type annotations.

When you're looking at a function or struct signature, you may see some special characters such as / and * in the parameter list. Here's an example:

def my_sort[
    # infer-only parameters
    dtype: DType,
    width: Int,
    //,
    # positional-only parameter
    values: SIMD[dtype, width],
    /,
    # positional-or-keyword parameter
    compare: def(Scalar[dtype], Scalar[dtype]) raises -> Int,
    *,
    # keyword-only parameter
    reverse: Bool = False,
]() -> SIMD[dtype, width]:

Here's a quick overview of the special characters in the parameter list:

    Double slash (//): parameters declared before the double slash are infer-only parameters.
    Slash (/): parameters declared before a slash are positional-only parameters. Positional-only and keyword-only parameters follow the same rules as positional-only and keyword-only arguments.
    A parameter name prefixed with a star, like *Types identifies a variadic parameter (not shown in the example above). Any parameters following the variadic parameter are keyword-only.
    Star (*): in a parameter list with no variadic parameter, a star by itself indicates that the following parameters are keyword-only parameters.
    An equals sign (=) introduces a default value for an optional parameter.

Parameters and generics

"Generics" refers to functions that can act on multiple types of values, or containers that can hold multiple types of values. For example, List, can hold different types of values, so you can have a list of Int values, or a list of String values.

In Mojo, generics use parameters to specify types. For example, List takes a type parameter, so a vector of integers is written List[Int]. So all generics use parameters, but not everything that uses parameters is a generic.

For example, the repeat[]() function in the previous section includes parameter of type Int, and an argument of type String. It's parameterized, but not generic. A generic function or struct is parameterized on type. For example, we could rewrite repeat[]() to take any type of argument that conforms to the Writable trait:

def repeat[MsgType: Writable, //, count: Int](msg: MsgType):
    comptime for i in range(count):
        print(msg)


def main() raises:
    # MsgType is always inferred, so first positional keyword `2` is
    # passed to `count`
    repeat[2](42)

42
42

This updated function takes any Writable type, so you can pass it an Int, String, or Bool value.

Note that there's a double-slash (//) in the parameter list after MsgType, to show that it's an infer-only parameter, so you don't need to specify it explicitly. Instead, the compiler sees that the msg argument is an Int and infers the type from the value.

Mojo's support for generics is still early. You can write generic functions like this using traits and parameters. You can also write generic collections like List and Dict. For more information, see the section on generics.
Parameterized structs

You can also add parameters to structs. You can use parameterized structs to build generic collections. For example, a generic array type might include code like this:

struct GenericArray[ElementType: Copyable & ImplicitlyDestructible]:
    var data: UnsafePointer[Self.ElementType, MutExternalOrigin]
    var size: Int

    def __init__(out self, var *elements: Self.ElementType):
        self.size = len(elements)
        self.data = alloc[Self.ElementType](self.size)
        for i in range(self.size):
            (self.data + i).init_pointee_move(elements[i].copy())

    def __del__(deinit self):
        for i in range(self.size):
            (self.data + i).destroy_pointee()
        self.data.free()

    def __getitem__(self, i: Int) raises -> ref[self] Self.ElementType:
        if i < self.size:
            return self.data[i]
        else:
            raise Error("Out of bounds")

This struct has a single parameter, ElementType, which is a placeholder for the data type you want to store in the array, sometimes called a type parameter. ElementType conforms to the Copyable trait and therefore to the Movable trait.

As with parameterized functions, you need to pass in parameter values when you use a parameterized struct. In this case, when you create an instance of GenericArray, you need to specify the type you want to store, like Int, or Float64. (This is a little confusing, because the parameter value you're passing in this case is a type. That's OK: a Mojo type is a valid compile-time value.)

You'll see that Self.ElementType is used throughout the struct where you'd usually see a type name. For example, as the formal type for the elements in the constructor, and the return type of the __getitem__() method.

Here's an example of using GenericArray:

var array = GenericArray(1, 2, 3, 4)
for i in range(array.size):
    end = ", " if i < array.size - 1 else "\n"
    print(array[i], end=end)

1, 2, 3, 4

A parameterized struct can use the Self type to represent a concrete instance of the struct (that is, with all its parameters specified). For example, you could add a static factory method to GenericArray with the following signature:

struct GenericArray[ElementType: Copyable & ImplicitlyDestructible]:
    ...

    @staticmethod
    def splat(count: Int, value: Self.ElementType) -> Self:
        # Create a new array with count instances of the given value

Here, Self is equivalent to writing GenericArray[Self.ElementType]. That is, you can call the splat() method like this:

GenericArray[Float64].splat(8, 0)

The method returns an instance of GenericArray[Float64].
Referencing struct parameters

As shown in the previous section, you reference a struct parameter using dot syntax, just like a struct method or field (for example, Self.ElementType).

This struct parameter access works anywhere, not just inside a struct's methods. You can access parameters as attributes on the type itself:

def on_type():
    print(SIMD[DType.float32, 2].size)  # prints 2

Or as attributes on an instance of the type:

def on_instance():
    var x = SIMD[DType.int32, 2](4, 8)
    print(x.dtype)  # prints int32

comptime members

You can also define comptime values as members of a struct or trait declaration:

struct Circle[radius: Float64]:
    comptime pi = 3.14159265359
    comptime circumference = 2 * Self.pi * Self.radius

These comptime members have a number of uses:

    Constant values specific to the type.
    Constant values calculated based on the struct's parameters.
    Associated types based on the struct's parameters.

The difference between parameters and comptime members is that parameter values are specified by the user, but comptime members represent either constant values or values derived from the input parameters.

A trait can declare a comptime member, which must be defined by all conforming structs.

Referencing comptime members works just like referencing struct parameters—you can reference a member using dot syntax (such as Self.IteratorType).
comptime members as enumerations

Some Mojo types use comptime members to express enumerations. For example, the following code defines a Sentiment type that defines comptime constants for different sentiment values:

@fieldwise_init
struct Sentiment(Equatable, ImplicitlyCopyable):
    var _value: Int

    comptime NEGATIVE = Sentiment(0)
    comptime NEUTRAL = Sentiment(1)
    comptime POSITIVE = Sentiment(2)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

def is_happy(s: Sentiment):
    if s == Sentiment.POSITIVE:
        print("Yes. 😀")
    else:
        print("No. ☹️")

This pattern provides a type-safe enumeration.

The DType struct implements a simple enum using comptime members like this. This allows clients to use values like DType.float32 in parameter expressions or runtime expressions.
comptime members as associated types

Associated types are a common use for comptime members. For example, a List[T] struct holds values of type T. The list's __iter__() method returns a list iterator that returns values of type T. List uses a comptime member, IteratorType, to define the type of the returned iterator.

The following code excerpt shows a simplified version of some of the List code, showing the List and its associated IteratorType:

@fieldwise_init
struct _ListIter[
    mut: Bool,
    //,
    T: Copyable,
    origin: Origin[mut],
](ImplicitlyCopyable, Iterable, Iterator):

    comptime Element = Self.T  # Required by the Iterator trait

    var index: Int
    var src: Pointer[List[Self.Element], Self.origin]

    # ... implementation omitted

struct List[T: Copyable](
    Boolable, Copyable, Defaultable, Iterable, Sized
):
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[iterable_mut]
    ]: Iterator = _ListIter[Self.T, iterable_origin]

    # ... code omitted

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return {0, Pointer(to=self)}

    # ... code omitted

The IteratorType member is parameterized on an origin, so it can represent both mutable and immutable iterators.
Struct methods

A struct's method can take its own parameters. For example, the SIMD.slice() method takes a size parameter:

var m = SIMD[DType.int32](1, 3, 5, 7)
var n = m.slice[2]()
print(n)  # prints [1, 3]

A struct's lifecycle methods (__init__() and __del__()) are an exception to this rule—they can't take parameters.
Case study: the SIMD type

For a real-world example of a parameterized type, let's look at the SIMD type from Mojo's standard library.

Single instruction, multiple data (SIMD) is a parallel processing technology built into many modern CPUs, GPUs, and custom accelerators. SIMD allows you to perform a single operation on multiple pieces of data at once. For example, if you want to take the square root of each element in an array, you can use SIMD to parallelize the work.

Processors implement SIMD using low-level vector registers in hardware that hold multiple instances of a scalar data type. To use the SIMD instructions on these processors, the data must be shaped into the proper SIMD width (data type) and length (vector size). Processors may support 512-bit or longer SIMD vectors, and support many data types from 8-bit integers to 64-bit floating point numbers, so it's not practical to define all of the possible SIMD variations.

Mojo's SIMD type (defined as a struct) exposes the common SIMD operations through its methods, and takes the SIMD data type and size values as parameters. This allows you to directly map your data to the SIMD vectors on any hardware.

Here's a cut-down (non-functional) version of Mojo's SIMD type definition:

struct SIMD[dtype: DType, size: Int]:
    var value: … # Some low-level MLIR stuff here

    # Create a new SIMD from a number of scalars
    def __init__(out self, *elems: SIMD[Self.dtype, 1]):  ...

    # Fill a SIMD with a duplicated scalar value.
    @staticmethod
    def splat(x: SIMD[Self.dtype, 1]) -> SIMD[Self.dtype, Self.size]: ...

    # Cast the elements of the SIMD to a different elt type.
    def cast[target: DType](self) -> SIMD[target, Self.size]: ...

    # Many standard operators are supported.
    def __add__(self, rhs: Self) -> Self: ...

So you can create and use a SIMD vector like this:

var vector = SIMD[DType.int16, 4](1, 2, 3, 4)
vector = vector * vector
for i in range(4):
    print(vector[i], end=" ")

1 4 9 16

As you can see, a simple arithmetic operator like * applied to a pair of SIMD vector operates on the corresponding elements in each vector.

Defining each SIMD variant with parameters is great for code reuse because the SIMD type can express all the different vector variants statically, instead of requiring the language to pre-define every variant.

Because SIMD is a parameterized type, the self argument in its functions carries those parameters—the full type name is SIMD[type, size]. Although it's valid to write this out (as shown in the return type of splat()), this can be verbose, so we recommend using the Self type (from PEP673) like the __add__ example does.
Using parameterized types and functions

You can use parameterized types and functions by passing values to the parameters in square brackets. For example, for the SIMD type above, dtype specifies the data type and size specifies the length of the SIMD vector (which must be a power of 2):

def main() raises:
    # Make a vector of 4 floats.
    var small_vec = SIMD[DType.float32, 4](1.0, 2.0, 3.0, 4.0)

    # Make a big vector containing 1.0 in float16 format.
    var big_vec = SIMD[DType.float16, 32](1.0)

    # Do some math and convert the elements to float32.
    var bigger_vec = (big_vec + big_vec).cast[DType.float32]()

    # You can write types out explicitly if you want of course.
    var bigger_vec2: SIMD[DType.float32, 32] = bigger_vec

    print("small_vec DType:", small_vec.dtype, "size:", small_vec.size)
    print(
        "bigger_vec2 DType:",
        bigger_vec2.dtype,
        "size:",
        bigger_vec2.size,
    )

small_vec DType: float32 size: 4
bigger_vec2 DType: float32 size: 32

Note that the cast() method also needs a parameter to specify the type you want from the cast (the method definition above expects a target parameter value). Thus, just as the SIMD struct is a generic type definition, the cast() method is a generic method definition. At compile time, the compiler creates a concrete version of the cast() method with the target parameter bound to DType.float32.

The code above shows the use of concrete types (that is, the parameters are all bound to known values). But the major power of parameters comes from the ability to define parameterized algorithms and types (code that uses the parameter values). For example, here's how to define a parameterized algorithm with Scalar that is datatype agnostic:

from std.math import sqrt

def rsqrt[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    return 1 / sqrt(x)

def main() raises:
    var v = Scalar[DType.float16](42)
    print(rsqrt(v))

0.154296875

Parameter inference

The Mojo compiler can often infer parameter values, so you don't always have to specify them. For example, in the previous section, this is how we called the parameterized rsqrt() function:

var v = Scalar[DType.float16](42)
print(rsqrt(v))

The compiler infers the dt parameter based on the type of the v value passed into it, as if you wrote rsqrt[DType.float16](v) explicitly. Figure 1 shows a mental model for how parameter inference works.

Figure 1. Parameter inference

Parameter inference can seem a little confusing: it might seem like the compiler is inferring compile-time parameter values from run-time argument values. But in fact it's inferring parameters from the statically-known types of the arguments.
Inference failures

If parameter inference fails, the compiler will report an error, usually "failed to infer parameter 'param_name'". Unfortunately, the compiler also sometimes reports this error incorrectly, for example, when the actual error is a type mismatch. In these cases, specifying the missing parameters explicitly will often allow Mojo to report the correct error.

Mojo can also infer the values of struct parameters from the arguments passed to a constructor or static method.

For example, consider the following struct:

struct One[Type: Writable & Copyable]:
    var value: Self.Type

    def __init__(out self, value: Self.Type):
        self.value = value.copy()


def use_one() raises:
    s1 = One(123)  # equivalent to One[Int](123)
    s2 = One("Hello")  # equivalent to One[String]("Hello")

Note that you can create an instance of One without specifying the Type parameter—Mojo can infer it from the value argument.

You can also infer parameters from a parameterized type passed to a constructor or static method:

struct Two[Type: Writable & Copyable]:
    var val1: Self.Type
    var val2: Self.Type

    def __init__(out self, one: One[Self.Type], another: One[Self.Type]):
        self.val1 = one.value.copy()
        self.val2 = another.value.copy()
        print(String(self.val1), String(self.val2))

    @staticmethod
    def fire(thing1: One[Self.Type], thing2: One[Self.Type]):
        print("🔥", String(thing1.value), String(thing2.value))

def use_two() raises:
    s3 = Two(One("infer"), One("me"))
    Two.fire(One(1), One(2))
    # Two.fire(One("mixed"), One(0)) # Error: parameter inferred to two
                                     # different values

use_two()

infer me
🔥 1 2

Two takes a Type parameter, and its constructor takes values of type One[Type]. When constructing an instance of Two, you don't need to specify the Type parameter, since it can be inferred from the arguments.

Similarly, the static fire() method takes values of type One[Type], so Mojo can infer the Type value at compile time. Note that passing two instances of One with different types doesn't work.

If you're familiar with C++, you may recognize this as similar to Class Template Argument Deduction (CTAD).
Parameter declarations

When you declare parameters on a struct or function, you have many of the same options as you have with arguments—you can define optional parameters with default values; keyword-only parameters; and variadic parameters.

In addition, you can define infer-only parameters, which provide a flexible way of defining dependencies between parameterized types.
Optional parameters and keyword parameters

Just as you can specify optional arguments in function signatures, you can also define an optional parameter by giving it a default value.

You can also pass parameters by keyword, just like you can use keyword arguments. For a function or struct with multiple optional parameters, using keywords allows you to pass only the parameters you want to specify, regardless of their position in the function signature.

For example, here's a function with two parameters, each with a default value:

def speak[a: Int = 3, msg: String = "woof"]():
    print(msg, a)


def use_defaults():
    speak()  # prints 'woof 3'
    speak[5]()  # prints 'woof 5'
    speak[7, "meow"]()  # prints 'meow 7'
    speak[msg="baaa"]()  # prints 'baaa 3'

Recall that when a parameterized function is called, Mojo can infer the parameter values. That is, it can determine its parameter values from the parameters attached to an argument. If the parameterized function also has a default value defined, then the inferred parameter value takes precedence.

For example, in the following code, we update the parameterized speak[]() function to take an argument with a parameterized type. Although the function has a default parameter value for a, Mojo instead uses the inferred a parameter value from the bar argument (as written, the default a value can never be used, but this is just for demonstration purposes):

@fieldwise_init
struct Bar[v: Int]:
    pass


def speak[a: Int = 3, msg: String = "woof"](bar: Bar[a]):
    print(msg, a)


def use_inferred():
    speak(Bar[9]())  # prints 'woof 9'

As mentioned above, you can also use optional parameters and keyword parameters in a struct:

struct KwParamStruct[greeting: String = "Hello", name: String = "🔥mojo🔥"]:
    def __init__(out self):
        print(Self.greeting, Self.name)

def use_kw_params():
    var a = KwParamStruct[]()                 # prints 'Hello 🔥mojo🔥'
    var b = KwParamStruct[name="World"]()     # prints 'Hello World'
    var c = KwParamStruct[greeting="Hola"]()  # prints 'Hola 🔥mojo🔥'

Mojo supports positional-only and keyword-only parameters, following the same rules as positional-only and keyword-only arguments.
Variadic parameters

Mojo also supports variadic parameters and runtime variadic packs:

struct MyTensor[*dimensions: Int]:
    pass

def sum_params[*values: Int]() -> Int:
    var sum = 0
    for value in values:
        sum += value
    return sum

def dump[*Ts: Writable](*args: *Ts):
    for arg in args:
        print(arg)

def forward[*Ts: Writable](*args: *Ts):
    dump(*args)

comptime assert TypeList[Trait=AnyType, Int, String]().contains[Int]

Variadic parameter lists can be indexed, iterated, and forwarded directly. Type variadics are exposed through `TypeList`.

Variadic keyword parameters (for example, **kwparams) are not supported yet.
Infer-only parameters

Sometimes you need to declare functions where parameters depend on other parameters. Because the signature is processed left to right, a parameter can only depend on a parameter earlier in the parameter list. For example:

def dependent_type[dtype: DType, value: Scalar[dtype]]():
    print("Value: ", value)
    print("Value is floating-point: ", dtype.is_floating_point())

dependent_type[DType.float64, Float64(2.2)]()

Value:  2.2000000000000002
Value is floating-point:  True

You can't reverse the position of the dtype and value parameters, because value depends on dtype. However, because dtype is a required parameter, you can't leave it out of the parameter list and let Mojo infer it from value:

dependent_type[Float64(2.2)]() # Error!

Infer-only parameters are a special class of parameters that are always either inferred from context or specified by keyword. Infer-only parameters are placed at the beginning of the parameter list, set off from other parameters by the // sigil:

def example[T: Copyable, //, list: List[T]]()

Transforming dtype into an infer-only parameter solves this problem:

def dependent_type[dtype: DType, //, value: Scalar[dtype]]():
    print("Value: ", value)
    print("Value is floating-point: ", dtype.is_floating_point())

dependent_type[Float64(2.2)]()

Value:  2.2000000000000002
Value is floating-point:  True

Because infer-only parameters are declared at the beginning of the parameter list, other parameters can depend on them, and the compiler will always attempt to infer the infer-only values from bound parameters or arguments.

There are sometimes cases where it's useful to specify an infer-only parameter by keyword. For example, the Span type is parameterized on origin:

struct Span[mut: Bool, //, T: Copyable, origin: Origin[mut]]:
    # ... implementation omitted

Here, the mut parameter is infer-only. The value is usually inferred when you create an instance of Span. Binding the mut parameter by keyword lets you define a Span that requires a mutable origin.

def mutate_span(span: Span[mut=True, Byte, ...]) raises:
    for i in range(0, len(span), 2):
        if i + 1 < len(span):
            span.swap_elements(i, i + 1)

If the compiler can't infer the value of an infer-only parameter, and it's not specified by keyword, compilation fails.

#### Origin-polymorphic trait methods

Trait methods that accept `UnsafePointer` (or other origin-parameterized types) often need to work with any caller-provided origin rather than requiring `MutAnyOrigin`. This avoids forcing callers to launder their pointer origins through `Int` and back.

The pattern is to add an `origin: MutOrigin` parameter that the compiler infers from the argument:

```mojo
trait Pool:
    def dispatch[Args: Copyable & ImplicitlyCopyable,
        kernel: def (Args) -> None, origin: MutOrigin](
        mut self, args: UnsafePointer[Args, origin], num_jobs: Int): ...

struct MyPool(Pool):
    def dispatch[Args: Copyable & ImplicitlyCopyable,
        kernel: def (Args) -> None, origin: MutOrigin](
        mut self, args: UnsafePointer[Args, origin], num_jobs: Int):
        for i in range(num_jobs):
            kernel((args + i)[])
```

Callers pass stack-local pointers directly without origin laundering:

```mojo
def use_pool[P: Pool](mut pool: P):
    var jobs = InlineArray[MyArgs, 128](uninitialized=True)
    jobs[0] = MyArgs(10, 20)
    # origin inferred from jobs — no MutAnyOrigin cast needed
    pool.dispatch[MyArgs, my_kernel](UnsafePointer(to=jobs[0]), 1)
```

Without origin polymorphism, the caller would need to erase and reconstruct the origin:

```mojo
# Bad: origin laundering through Int
pool.dispatch[MyArgs, my_kernel](
    UnsafePointer[MyArgs, MutAnyOrigin](unsafe_from_address=Int(UnsafePointer(to=jobs[0]))), 1)
```

Use `MutOrigin` when the data is always mutable (the common case for dispatch buffers). For read-only access, use the `is_mutable: Bool, //, origin: Origin[mut=is_mutable]` pattern to accept both mutable and immutable origins.

Parameter expressions are just Mojo code

A parameter expression is any code expression (such as a+b) that occurs where a parameter is expected. Parameter expressions support operators and function calls, just like runtime code, and all parameter types use the same type system as the runtime program (such as Int and DType).

Because parameter expressions use the same grammar and types as runtime Mojo code, you can use many "dependent type" features. For example, you might want to define a helper function to concatenate two SIMD vectors:

def concat[
    dtype: DType, ls_size: Int, rh_size: Int, //
](lhs: SIMD[dtype, ls_size], rhs: SIMD[dtype, rh_size]) -> SIMD[
    dtype, ls_size + rh_size
]:
    var result = SIMD[dtype, ls_size + rh_size]()

    comptime for i in range(ls_size):
        result[i] = lhs[i]

    comptime for j in range(rh_size):
        result[ls_size + j] = rhs[j]
    return result

Note that the resulting length is the sum of the input vector lengths, and this is expressed with a simple + operation.
Powerful compile-time programming

While simple expressions are useful, sometimes you want to write imperative compile-time logic with control flow. You can even do compile-time recursion. For instance, here is an example "tree reduction" algorithm that sums all elements of a vector recursively into a scalar:

def slice[
    dtype: DType, size: Int, //
](x: SIMD[dtype, size], offset: Int) -> SIMD[dtype, size // 2]:
    comptime new_size = size // 2
    var result = SIMD[dtype, new_size]()
    for i in range(new_size):
        result[i] = SIMD[dtype, 1](x[i + offset])
    return result


def reduce_add(x: SIMD) -> Int:
    comptime if x.size == 1:
        return Int(x[0])
    elif x.size == 2:
        return Int(x[0]) + Int(x[1])

    # Extract the top/bottom halves, add them, sum the elements.
    comptime half_size = x.size // 2
    var lhs = slice(x, 0)
    var rhs = slice(x, half_size)
    return reduce_add(lhs + rhs)


def main() raises:
    var x = SIMD[DType.int, 4](1, 2, 3, 4)
    print(x)
    print("Elements sum:", reduce_add(x))

[1, 2, 3, 4]
Elements sum: 10

This makes use of the comptime if statement, which is an if statement that runs at compile-time. It requires that its condition be a valid parameter expression, and ensures that only the live branch of the if statement is compiled into the program. This is similar to use of the comptime for loop shown earlier.
Parameterized comptime values

A parameterized comptime value is a compile-time expression that takes a list of parameters and returns a compile-time constant value:

comptime AddOne[a: Int] : Int = a + 1

comptime nine = AddOne[8]

As you can see in the previous example, a parameterized comptime value is a little like a compile-time-only function. A regular function or method can also be invoked at compile time:

def add_one(a: Int) -> Int:
    return a + 1

comptime ten = add_one(9)

A major difference between a function and a parameterized comptime value is that the value of a comptime expression can be a type, while a function can't return a type as a value.

# Does not work—-dynamic type values not permitted
def int_type() -> AnyType:
    return Int

# Works
comptime IntType = Int

Because a comptime value can be a type, you can use parameterized comptime values to express new types:

comptime TwoOfAKind[dt: DType] = SIMD[dt, 2]
twoFloats = TwoOfAKind[DType.float32](1.0, 2.0)

comptime StringKeyDict[ValueType: Copyable & ImplicitlyDestructible] = Dict[String, ValueType]
var b: StringKeyDict[UInt8] = {"answer": 42}

Parameterized comptime declarations support the same features as parameterized structs or functions: infer-only parameters, keyword-only and optional parameters, automatic parameterization, and so on.

def main():
    comptime Floats[size: Int, half_width: Bool = False] = SIMD[
        (DType.float16 if half_width else DType.float32), size
    ]
    var floats = Floats[2](6.0, 8.0)
    var half_floats = Floats[2, True](10.0, 12.0)

Fully-bound, partially-bound, and unbound types

A parameterized type with its parameters specified is said to be fully-bound. That is, all of its parameters are bound to values. As mentioned before, you can only instantiate a fully-bound type (sometimes called a concrete type).

However, parameterized types can be unbound or partially bound in some contexts. For example, you can use comptime to create a shorthand for a partially-bound type to create a new type that requires fewer parameters:

comptime StringKeyDict = Dict[String, _]
var b: StringKeyDict[UInt8] = {"answer": 42}

Here, StringKeyDict is a shorthand for a Dict that takes String keys. The underscore _ in the parameter list indicates that the second parameter, V (the value type), is unbound. You specify the V parameter later, when you use StringKeyDict.
Partially-bound types versus parameterized comptime values

You may notice that this example is very similar to an example in the section on parameterized comptime values. For simple type shorthands like this, you can use either a partially-bound type or a parameterized comptime value. Parameterized comptime values provide a more flexible way to define named type shorthands, since you can define the order of the parameters, add default values, and so on.

Partially-bound and unbound types can provide a handy shortcut when defining parameterized functions and comptime values, called automatic parameterization.

You can also use partially-bound types as the type bound for an argument or parameter.

For example, given the following type:

struct MyType[s: String, i: Int, i2: Int, b: Bool = True]:
    pass

It can appear in code in the following forms:

    Fully bound, with all of its parameters specified:

    def my_fn1(m1: MyType["Hello", 3, 4, True]) raises:
        pass

    Partially bound, with some but not all of its parameters specified:

    def my_fn2(m2: MyType["Hola", _, _, True]) raises:
        pass

    Unbound, with no parameters specified:

    def my_fn3(m3: MyType[_, _, _, _]) raises:
        pass

You can also use three dots (...) to unbind an arbitrary number of parameters at the end of a parameter list (including any keyword-only parameters):

# These two types are equivalent
MyType["Hello", ...]
MyType["Hello", _, _, _]

When a parameter is explicitly unbound with the _, or ... expressions, you must specify a value for that parameter to use the type. The default values of explicitly unbound parameters are ignored.

Partially-bound and unbound parameterized types can be used in some contexts where the missing (unbound) parameters will be supplied later—such as in comptime values and automatically parameterized functions.
Omitted parameters

Mojo also supports an alternate format for unbound parameters where parameters are simply omitted from the expression:

@fieldwise_init
struct MyComplicatedType[a: Int = 7, /, b: Int = 8, *, c: Int, d: Int = 9]:
    pass

# Unbound
def my_func(t: MyComplicatedType):
    pass

This is equivalent to def my_func(t: MyComplicatedType[...]): pass. That is, all parameters (positional-only, positional-or-keyword, keyword-only) are unbound and their default values (if any) ignored.

Note that when an argument type is partially bound, default values will be bound:

# Partially bound
MyComplicatedType[1]
# Equivalent to
MyComplicatedType[1, 8, c=_, d=9]  # Uses default values for `b` and `d`.

This behavior with omitted parameters is currently supported for backwards compatibility. We intend to reconcile the behavior of omitted parameters and explicitly unbound parameters in the future.
Automatic parameterization

Mojo supports "automatic" parameterization of functions and parameterized comptime values. If a function argument type or parameter type is partially-bound or unbound, the unbound parameters are automatically added as parameters on the function. This is easier to understand with an example:

def print_params(vec: SIMD):
    print(vec.dtype)
    print(vec.size)

var v = SIMD[DType.float64, 4](1.0, 2.0, 3.0, 4.0)
print_params(v)

float64
4

In the above example, the print_params() function is automatically parameterized. The vec argument takes an argument of type SIMD[...]. This is an unbound parameterized type—that is, it doesn't specify any parameter values for the type. Mojo treats the unbound parameters on vec as infer-only parameters on the function. This is roughly equivalent to the following code:

def print_params2[t: DType, s: Int, //](vec: SIMD[t, s]):
    print(vec.dtype)
    print(vec.size)

When you call print_params() you must pass it a concrete instance of the SIMD type—that is, one with all of its parameters specified, like SIMD[DType.float64, 4]. The Mojo compiler infers the parameter values from the input argument.

With a manually parameterized function, you can access the parameters by name (for example, t and s in the previous example), which is not an option in an automatically parameterized function.

However, you can always access a type's parameters and comptime members using dot syntax (for example, vec.dtype), as described in Referencing struct parameters. This ability to access a type's parameters and comptime members is not specific to automatically parameterized functions, you can use it anywhere.

You can even use this syntax in the function's signature to define a function's arguments and return type based on an argument's parameters or comptime members.

For example, if you want your function to take two SIMD vectors with the same type and size, you can write code like this:

def interleave(v1: SIMD, v2: type_of(v1)) -> SIMD[v1.dtype, v1.size*2]:
    var result = SIMD[v1.dtype, v1.size*2]()
    for i in range(v1.size):
        result[i*2] = SIMD[v1.dtype, 1](v1[i])
        result[i*2+1] = SIMD[v1.dtype, 1](v2[i])
    return result

var a = SIMD[DType.int16, 4](1, 2, 3, 4)
var b = SIMD[DType.int16, 4](0, 0, 0, 0)
var c = interleave(a, b)
print(c)

[1, 0, 2, 0, 3, 0, 4, 0]

As shown in the example, you can use the magic type_of(x) call if you just want to match the type of an argument. In this case, it's more convenient and compact than writing the equivalent SIMD[v1.dtype, v1.size].
Automatic parameterization of parameters

You can also take advantage of automatic parameterization in the parameter list of a function or parameterized comptime value. For example:

def foo[value: SIMD]():
    pass

# Equivalent to:
def foo[dtype: DType, size: Int, //, value: SIMD[dtype, size]]():
    pass

Here's another example using a parameterized comptime value:

comptime Foo[S: SIMD] = Bar[S]

# Equivalent to:
comptime Foo[dtype: DType, size: Int, //, S: SIMD[dtype, size]] = Bar[S]

Automatic parameterization with partially-bound types

Mojo also supports automatic parameterization: with partially-bound parameterized types (that is, types with some but not all of the parameters specified).

For example, suppose we have a Fudge struct with three parameters:

@fieldwise_init
struct Fudge[sugar: Int, cream: Int, chocolate: Int = 7](Writable):
    pass

We can write a function that takes a Fudge argument with just one bound parameter (it's partially bound):

def eat(f: Fudge[5, ...]):
    print("Ate " + String(f))

The eat() function takes a Fudge struct with the first parameter (sugar) bound to the value 5. The second and third parameters, cream and chocolate are unbound.

The unbound cream and chocolate parameters become implicit parameters on the eat function. In practice, this is roughly equivalent to writing:

def eat[cr: Int, ch: Int](f: Fudge[5, cr, ch]):
    print("Ate", String(f))

In both cases, we can call the function by passing in an instance with the cream and chocolate parameters bound:

eat(Fudge[5, 5, 7]())
eat(Fudge[5, 8, 9]())

Ate Fudge (5,5,7)
Ate Fudge (5,8,9)

If you try to pass in an argument with a sugar value other than 5, compilation fails, because it doesn't match the argument type:

eat(Fudge[12, 5, 7]())
# ERROR: invalid call to 'eat': argument #0 cannot be converted from 'Fudge[12, 5, 7]' to 'Fudge[5, 5, 7]'

You can also explicitly unbind individual parameters. This gives you more freedom in specifying unbound parameters.

For example, you might want to let the user specify values for sugar and chocolate, and leave cream constant. To do this, replace each unbound parameter value with a single underscore (_):

def devour(f: Fudge[_, 6, _]):
    print("Devoured",  String(f))

Again, the unbound parameters (sugar and chocolate) are added as implicit parameters on the function. This version is roughly equivalent to the following code, where these two values are explicitly bound to the input parameters, su and ch:

def devour[su: Int, ch: Int](f: Fudge[su, 6, ch]):
    print("Devoured", String(f))

You can also specify parameters by keyword, or mix positional and keyword parameters, so the following function is roughly equivalent to the previous one: the first parameter, sugar is explicitly unbound with the underscore character. The chocolate parameter is unbound using the keyword syntax, chocolate=_. And cream is explicitly bound to the value 6:

def devour(f: Fudge[_, chocolate=_, cream=6]):
    print("Devoured", String(f))

All three versions of the devour() function work with the following calls:

devour(Fudge[3, 6, 9]())
devour(Fudge[4, 6, 8]())

Devoured Fudge (3,6,9)
Devoured Fudge (4,6,8)

The rebind() builtin

One of the consequences of Mojo not performing function instantiation in the parser like C++ is that Mojo cannot always figure out whether some parameterized types are equal and complain about an invalid conversion. This typically occurs in static dispatch patterns. For example, the following code won't compile:

def take_simd8(x: SIMD[DType.float32, 8]):
    pass

def generic_simd[nelts: Int](x: SIMD[DType.float32, nelts]):
    comptime if nelts == 8:
        take_simd8(x)

The parser will complain:

error: invalid call to 'take_simd8': argument #0 cannot be converted from
'SIMD[f32, nelts]' to 'SIMD[f32, 8]'
        take_simd8(x)
        ~~~~~~~~~~^~~

This is because the parser fully type-checks the function without instantiation, and the type of x is still SIMD[f32, nelts], and not SIMD[f32, 8], despite the static conditional. The remedy is to manually "rebind" the type of x, using the rebind builtin, which inserts a compile-time assert that the input and result types resolve to the same type after function instantiation:

def take_simd8(x: SIMD[DType.float32, 8]):
    pass

def generic_simd[nelts: Int](x: SIMD[DType.float32, nelts]):
    comptime if nelts == 8:
        take_simd8(rebind[SIMD[DType.float32, 8]](x))
        
Traits

A trait is a set of requirements that a type must implement. You can think of it as a contract: a type that conforms to a trait guarantees that it implements all of the features of the trait.

Traits are similar to Java interfaces, C++ concepts, Swift protocols, and Rust traits. If you're familiar with any of those features, Mojo traits solve the same basic problem.

You've probably already seen some traits, like Copyable and Movable, used in example code. This section describes how traits work, how to use existing traits, and how to define your own traits.
Background

In dynamically-typed languages like Python, you don't need to explicitly declare that two classes are similar. This is easiest to show by example:
🐍 Python

class Duck:
    def quack(self):
        print("Quack.")

class StealthCow:
    def quack(self):
        print("Moo!")

def make_it_quack(maybe_a_duck):
    try:
        maybe_a_duck.quack()
    except:
        print("Not a duck.")

make_it_quack(Duck())
make_it_quack(StealthCow())

The Duck and StealthCow classes aren't related in any way, but they both define a quack() method, so they work the same in the make_it_quack() function. This works because Python uses dynamic dispatch—it identifies the methods to call at runtime. So make_it_quack() doesn't care what types you're passing it, only the fact that they implement the quack() method.

In a statically-typed environment, this approach doesn't work: Mojo functions require you to specify the type of each argument. If you wanted to write this example in Mojo without traits, you'd need to write a function overload for each input type.
🔥 Mojo

@fieldwise_init
struct Duck(Copyable):
    def quack(self):
        print("Quack")

@fieldwise_init
struct StealthCow(Copyable):
    def quack(self):
        print("Moo!")

def make_it_quack(definitely_a_duck: Duck):
    definitely_a_duck.quack()

def make_it_quack(not_a_duck: StealthCow):
    not_a_duck.quack()

make_it_quack(Duck())
make_it_quack(StealthCow())

Quack
Moo!

This isn't too bad with only two types. But the more types you want to support, the less practical this approach is.

You might notice that the Mojo versions of make_it_quack() don't include the try/except statement. We don't need it because Mojo's static type checking ensures that you can only pass instances of Duck or StealthCow into the make_it_quack() function.
Using traits

Traits solve this problem by letting you define a shared set of behaviors that types can implement. Then you can write a function that depends on the trait, rather than individual types. As an example, let's update the make_it_quack() example using traits. This will involve three steps:

    Defining a new Quackable trait.
    Adding the trait to the Duck and StealthCow structs.
    Updating the make_it_quack() function to depend on the trait.

Defining a trait

The first step is defining a trait that requires a quack() method:

trait Quackable:
    def quack(self):
        ...

A trait looks a lot like a struct, except it's introduced by the trait keyword. Note that the quack() method signature is followed by three dots (...). This indicates it isn't implemented within the trait. In this example, quack is a required method and must be implemented by any conforming struct.

A trait can supply a default implementation, so conforming structs don't need to implement the method themselves. You can provide a full implementation or use the pass keyword. Using pass creates a no-op method that does nothing. In your conforming struct, you can choose to override this default implementation:

trait Quackable:
    def quack(self):
        pass

For more information, see Default method implementations.

A trait can also include comptime members—compile-time constant values that must be defined by conforming structs. comptime members are useful for writing traits that describe generic types. For more information, see comptime members for generics.
Adding traits to structs

Next, we need some structs that conform to the Quackable trait. Since the Duck and StealthCow structs above already implement the quack() method, all we need to do is add the Quackable trait to the traits it conforms to (in parenthesis, after the struct name).

(If you're familiar with Python, this looks just like Python's inheritance syntax.)

@fieldwise_init
struct Duck(Copyable, Quackable):
    def quack(self):
        print("Quack")

@fieldwise_init
struct StealthCow(Copyable, Quackable):
    def quack(self):
        print("Moo!")

The struct needs to implement any methods that are declared in the trait. The compiler enforces conformance: if a struct says it conforms to a trait, it must implement everything required by the trait, or the code won't compile.
Using a trait as a type bound

Finally, you can define a function that takes a Quackable like this:

def make_it_quack[DuckType: Quackable](maybe_a_duck: DuckType):
    maybe_a_duck.quack()

Or using the shorthand form:

def make_it_quack2(maybe_a_duck: Some[Quackable]):
    maybe_a_duck.quack()

This syntax may look a little unfamiliar if you haven't dealt with Mojo parameters before. What the first signature means is that maybe_a_duck is an argument of type DuckType, where DuckType is a type that must conform to the Quackable trait. Quackable is called the type bound for DuckType.

The Some[Quackable] form expresses the same idea: the type of maybe_a_duck is some concrete type that conforms to the trait Quackable.

Both forms work the same, except that the first form explicitly names the type value. This can be useful, for example, if you want to take two values of the same type:

def take_two_quackers[DuckType: Quackable](quacker1: DuckType, quacker2: DuckType):
    pass

Putting it all together

Using the function is simple enough:

make_it_quack(Duck())
make_it_quack(StealthCow())

Quack
Moo!

Note that you don't need the square brackets when you call make_it_quack(): the compiler infers the type of the argument, and ensures the type has the required trait.

One limitation of traits is that you can't add traits to existing types. For example, if you define a new Numeric trait, you can't add it to the standard library Float64 and Int types. However, the standard library already includes quite a few traits, and we'll be adding more over time.
Traits can require static methods

In addition to regular instance methods, traits can specify required static methods.

trait HasStaticMethod:
    @staticmethod
    def do_stuff(): ...

def fun_with_traits[type: HasStaticMethod]():
    type.do_stuff()

Default method implementations

Often, some or all of the structs that conform to a given trait can use the same implementation for a given required method. In this case, the trait can include a default implementation. A conforming struct can still define its own version of the method, overriding the default implementation. But if the struct doesn't define its own version, it automatically inherits the default implementation.

Defining a default implementation for a trait looks the same as writing a method for a struct:

trait DefaultQuackable:
    def quack(self):
        print("Quack")


@fieldwise_init
struct DefaultDuck(Copyable, DefaultQuackable):
    pass

When looking at the API doc for a standard library trait, you'll see methods that you must implement listed as required methods, and methods that have default implementations listed as provided methods.

The Equatable trait is a good example of the use case for default implementations. The trait includes two methods: __eq__() (corresponding to the == operator) and __ne__() (corresponding to the != operator). Every type that conforms to Equatable needs to define the __eq__() method for itself, but the trait supplies a default implementation for __ne__(). Given an __eq__() method, the definition of __ne__() is trivial for most types:

def __ne__(self, other: Self) -> Bool:
    return not self.__eq__(other)

Trait compositions

Mojo uses `def` for all function declarations and function type positions. Named compile-time type shorthands are written with `comptime`.

You can compose traits using the & sigil. This lets you define new traits that are simple combinations of other traits. You can use a trait composition anywhere that you'd use a single trait:

trait Flyable:
    def fly(self): ...

def quack_and_go[type: Quackable & Flyable](quacker: type):
    quacker.quack()
    quacker.fly()

@fieldwise_init
struct FlyingDuck(Copyable, Quackable, Flyable):
    def quack(self):
        print("Quack")

    def fly(self):
        print("Whoosh!")

You can also use the comptime keyword to create a shorthand name for a trait composition:

comptime DuckLike = Quackable & Flyable

struct ToyDuck(DuckLike):
    # ... implementation omitted

You can also compose traits using refinement, by defining a new, empty trait like this:

trait DuckTrait(Quackable, Flyable):
    pass

However, this is less flexible than using a trait composition and not recommended. The difference is that using the trait keyword defines a new, named trait. For a struct to conform to this trait, you need to explicitly include it in the struct's signature. On the other hand, the DuckLike comptime value represents a composition of two separate traits, Quackable and Flyable, and anything that conforms to those two traits conforms to DuckLike. For example, consider the FlyingDuck type shown above:

struct FlyingDuck(Copyable, Quackable, Flyable):
    # ... etc

Because FlyingDuck conforms to both Quackable and Flyable, it also conforms to the DuckLike trait composition. But it doesn't conform to DuckTrait, since it doesn't include DuckTrait in its list of traits.
Trait refinement

Traits refine other traits. A trait that refines another trait includes all of the requirements declared by the original trait. For example:

trait Animal:
    def make_sound(self):
        ...

# Bird refines Animal
trait Bird(Animal):
    def fly(self):
        ...

Since Bird refines Animal, a struct that conforms to the Bird trait must implement both make_sound() and fly(). And since every Bird conforms to Animal, a struct that conforms to Bird can be passed to any function that requires an Animal.

You can define a trait as a refinement of multiple traits by listing them in parentheses. This can be either a comma-separated list of traits or a trait composition. For example, you might define a NamedAnimal trait that combines the requirements of the Animal trait and a new Named trait:

trait Named:
    def get_name(self) -> String:
        ...

trait NamedAnimal(Animal, Named):
    def emit_name_and_sound(self):
        ...

Refinement is useful when you're creating a new trait that adds additional requirements. If you simply want to express the union of two or more traits, use a simple trait composition instead:

comptime NamedAnimal = Animal & Named

Traits and lifecycle methods

Traits can specify required lifecycle methods, including constructors, copy constructors and move constructors.

For example, the following code creates a MassProducible trait. A MassProducible type has a default (no-argument) constructor and can be moved. It uses two built-in traits: Defaultable, which requires a default (no-argument) constructor, and Movable, which requires the type to have a move constructor.

The factory[]() function returns a newly-constructed instance of a MassProducible type. The following example shows the definitions of the Defaultable and Movable traits in comments for reference:

# trait Defaultable
#     def __init__(out self): ...

# trait Movable
#     def __init__(out self, *, deinit take: Self): ...

comptime MassProducible = Defaultable & Movable

def factory[type: MassProducible]() -> type:
    return type()

struct Thing(MassProducible):
    var id: Int

    def __init__(out self):
        self.id = 0

    def __init__(out self, *, deinit take: Self):
        self.id = take.id

var thing = factory[Thing]()

Register passable types and the RegisterPassable trait

"Register passable" tells Mojo that a type should be passed in machine registers such as a CPU register. This means the type is always passed by value. For data types like an integer or floating-point number, this is much more efficient than storing values in stack memory.

Mojo supports two forms of register passable types.

The standard register passable type has the following capabilities and restrictions:

    They have normal lifecycles and can implement or override lifecycle methods.
    Every field must either be register passable or trivially register passable.
    The type must be Movable.
    Their self pointer is neither stable nor predictable. They can move in memory at any time so they aren't suitable for types that rely on pointer identity.
    When used as arguments or results, they can be exposed directly to C and C++ through foreign function interfaces (FFI) and don't need to be passed by pointer.
    They can't override the default move constructor. This guarantees moves are always side-effect-free.

You can create custom register-passable types by conforming a type to the RegisterPassable trait.

Trivial register types are typically data types. They include:

    Arithmetic types. This includes types such as Int, Int32, Bool, Float64 etc.
    Pointers. The address value is trivial, not the data being pointed to.
    Arrays of other trivial types. SIMD is a good example.

These types are provided by Mojo's standard library.

A trivially register passable type has the following capabilities and restrictions:

    They do not use lifecycle methods.
    Every field must be trivially register passable.
    The type must be Copyable.
    They use a special trivial (no-op) destructor. They are trivially movable (via Copyable, which refines Movable), trivially copyable, and trivially destructible.

Built-in traits

The Mojo standard library includes many traits. They're implemented by a number of standard library types, and you can also implement these on your own types. These standard library traits include:

    Absable
    AnyType
    Boolable
    Comparable
    Copyable
    Defaultable
    Hashable
    ImplicitlyDestructible
    Indexer
    Intable
    IntableRaising
    KeyElement
    Movable
    PathLike
    Powable
    Sized
    Roundable
    Writable
    Writer

The API reference docs linked above include usage examples for each trait. The following sections discuss a few of these traits.
The Sized trait

The Sized trait identifies types that have a measurable length, like strings and arrays.

Specifically, Sized requires a type to implement the __len__() method. This trait is used by the built-in len() function. For example, if you're writing a custom list type, you could implement this trait so your type works with len():

struct MyList(Copyable, Sized):
    var size: Int
    # ...

    def __init__(out self):
        self.size = 0

    def __len__(self) -> Int:
        return self.size

print(len(MyList()))

0

The Intable and IntableRaising traits

The Intable trait identifies a type that can be converted to Int. The IntableRaising trait describes a type can be converted to an Int, but the conversion might raise an error.

Both of these traits require the type to implement the __int__() method. For example:

@fieldwise_init
struct IntLike(Intable):
    var i: Int

    def __int__(self) -> Int:
        return self.i

value = IntLike(42)
print(Int(value) == 42)

True

The Writable trait

The Writable trait describes a type that can produce a human-readable text representation by writing to a Writer object. The print() function requires that its arguments conform to the Writable trait. This enables efficient stream-based writing by default, avoiding unnecessary intermediate String heap allocations.

The Writable trait requires a type to implement a write_to() method. This method receives a mut writer: Some[Writer] argument. A generic Writer that represents anything that can be written to, such as a String buffer, a file, or a network stream. You invoke the Writer instance's write() method to write a sequence of Writable arguments constituting the string representation of your type.

An important feature of the Writer trait is that it only accepts valid UTF-8 text. This means that when you write data to a Writer, you can only use StringSlice values (via the write_string() method) or other Writable types; you can't write arbitrary bytes. This guarantee ensures that types like String can safely implement the Writer trait without risking corruption of their internal UTF-8 data.

Writable also provides a write_repr_to() method for producing the "official" string representation of a type—the kind you would see when calling repr() or using the {!r} format specifier. If at all possible, this should look like a valid Mojo expression that could be used to recreate a struct instance with the same value. write_repr_to() has a default implementation that uses reflection, so you only need to override it if your type needs a distinct debug representation.

Here is a simple example of a type that implements Writable with a custom write_repr_to():

@fieldwise_init
struct Dog(Copyable, Writable):
    var name: String
    var age: Int

    # Allows the type to be written into any `Writer`
    def write_to(self, mut writer: Some[Writer]):
        t"Dog({self.name}, {self.age})".write_to(writer)

    # Official representation when calling `repr`
    def write_repr_to(self, mut writer: Some[Writer]):
        t"Dog(name={self.name}, age={self.age})".write_to(writer)

dog = Dog("Rex", 5)
print(repr(dog))
print(dog)

Dog(name=Rex, age=5)
Dog(Rex, 5)

Lifetime management traits

Mojo provides two core traits for managing value lifetimes:

    AnyType: The base trait that all types extend. Types conforming only to AnyType may require explicit destruction.

    ImplicitlyDestructible: Types that can be automatically destroyed by calling __del__() when their lifetime ends.

For detailed information about value destruction, explicit destruction with the @explicit_destroy decorator, and when to use each approach, see Death of a value.
Generic structs with traits

You can also use traits when defining a generic container. A generic container is a container (for example, an array or hashmap) that can hold different data types. In a dynamic language like Python it's easy to add different types of items to a container. But in a statically-typed environment, the compiler needs to be able to identify the types at compile time. For example, if the container needs to copy a value, the compiler needs to verify that the type can be copied.

The List type is an example of a generic container. A single List can only hold a single type of data. The list elements must conform to the Copyable trait:

struct List[T: Copyable]:

For example, you can create a list of integer values like this:

var list: List[Int]
list = [1, 2, 3, 4]
for i in range(len(list)):
    print(list[i], end=" ")

1 2 3 4

You can use traits to define requirements for elements that are stored in a container. For example, List requires elements that can be moved and copied. To store a struct in a List, the struct needs to conform to the Copyable trait, which requires a copy constructor and a move constructor.

Building generic containers is an advanced topic. For an introduction, see the section on parameterized structs.
comptime members for generics

In addition to methods, a trait can include comptime members, which must be defined by any conforming struct. For example:

trait Repeater:
    comptime count: Int

An implementing struct must define a concrete constant value for the comptime member, using any compile-time parameter value. For example, it can use a literal constant or a compile-time expression, including one that uses the struct's parameters.

struct Doublespeak(Repeater):
    comptime count: Int = 2

struct Multispeak[verbosity: Int](Repeater):
    comptime count: Int = Self.verbosity * 2 + 1

The Doublespeak struct has a constant value for count, but the Multispeak struct lets the user set the value using a parameter:

repeater = Multispeak[12]()

Note that the field is named count, and the Multispeak parameter is named verbosity. Parameters and comptime members are in the same namespace, so the parameter can't have the same name as the comptime member.

comptime members are most useful for writing traits for generic types. For example, imagine that you want to write a trait that describes a generic stack data structure that stores elements that conform to the Copyable trait.

By adding the element type as a comptime member on the trait, you can specify generic methods on the trait:

trait Stacklike:
    comptime EltType: Copyable

    def push(mut self, var item: Self.EltType):
        ...

    def pop(mut self) -> Self.EltType:
        ...

The following struct implements the Stacklike trait using a List as the underlying storage:

struct MyStack[type: Copyable & ImplicitlyDestructible](Stacklike):
    """A simple Stack built using a List."""
    comptime EltType = Self.type
    comptime list_type = List[Self.EltType]

    var list: Self.list_type

    def __init__(out self):
        self.list = Self.list_type()

    def push(mut self, var item: Self.EltType):
        self.list.append(item^)

    def pop(mut self) -> Self.EltType:
        return self.list.pop()

    def dump[
        WritableEltType: Writable & Copyable
    ](self: MyStack[WritableEltType]):
        print("[", end="")
        for item in self.list:
            print(item, end=", ")
        print("]")

The MyStack type adds a dump() method that prints the contents of the stack. Because a struct that conforms to Copyable is not necessarily printable, MyStack uses conditional conformance to define a dump() method that works as long as the element type is writable.

The following code exercises this new trait by defining a generic method, add_to_stack() that adds an item to any Stacklike type.

def add_to_stack[S: Stacklike](mut stack: S, var item: S.EltType) raises:
    stack.push(item^)

def main() raises:
    s = MyStack[Int]()
    add_to_stack(s, 12)
    add_to_stack(s, 33)
    s.dump()             # [12, 33, ]
    print(s.pop())       # 33
    
Generics

Generics let you write code that builds and runs across many types. This matters because you don't have to write type-specific versions of the same logic over and over again. Instead of branching on "this type" versus "that one," you can collapse nearly identical implementations into a single, maintainable solution.

You don't duplicate work. You build unified solutions that automatically adapt at compile time to the types you already use and the types you'll use in the future. It doesn't matter whether those types come from your own code or from libraries you didn't write.

Generics work through traits to make this possible. Traits define what a type must be able to do, and generics let you write code that operates over any type that meets those trait requirements. One type, method, or function can then work across many source types, eliminating redundant "almost the same" implementations and centralizing fixes in one place.

At their core, generics give you the power to make code work with any type that fits its defined requirements. You express your intent once and apply it broadly, without sacrificing correctness or clarity.

Generic code creates custom compiled code for each concrete type it's used with. While this enables performance enhancements and supports static checking, this will increase both compile times and code size.
Setting generic parameter bounds

Bounds specify what a type must be able to do. They're requirements written as traits that restrict which types can be used with generic code.

You must always restrict generic parameters with trait bounds. Traits define the vocabulary generic code relies on. They specify available operations, guarantees, and associated types (comptime aliases for related types). This vocabulary allows generic code to express behavior, access related types, and gives the compiler enough information to reason about correctness. This lets it emit warnings about missing elements, whether those elements relate to type capabilities or lifetime management.

    With generics, the most permissive trait bound is AnyType. It places no behavioral requirements on a type and serves as the least restrictive bound available.

    ImplicitlyDestructible is another baseline trait. It is the base trait for types that require lifetime management using destructors. Any type that needs cleanup when it goes out of scope should implement this trait. Generic code that stores or owns values often relies on it.

Bounds are what make generic code workable. Without them, the compiler can't allow useful operations, and functions and methods can't do anything meaningful.

Examples throughout this page use trait composition (like Equatable & Copyable) to spell out the features their algorithms depend on. Each trait defines required API elements that types used from call sites must provide.
Basic generics: compare two lists

Consider comparing two lists to test if they contain the same values in the same order.

You could write a separate implementation for each element type used in lists. Or you could write a single generic function that works for any list whose elements can be compared.

Here's a concrete version that's only useful for Integers:

def all_equal_int(ref lhs: List[Int], ref rhs: List[Int]) -> Bool:
    if len(lhs) != len(rhs): return False

    for left, right in zip(lhs, rhs):
        if left != right:
            return False
    return True

Unlike this function, a generic version doesn't care which element type it uses. It only demands the capabilities the algorithm uses: elements must be comparable for equality. They're used as loop values, so they must be copyable.

Given those requirements, your function can walk both lists element-by-element, just like the Integer version.

def all_equal[
    T: Equatable & Copyable
](ref lhs: List[T], ref rhs: List[T]) -> Bool:
    if len(lhs) != len(rhs): return False

    for left, right in zip(lhs, rhs):
        if left != right:
            return False
    return True

Both implementations check the list lengths, returning False if they're not the same. Then they walk the list, using an early return of False on the first mismatch. If the function makes it to the end, it returns True. No differences were found.
How this works

The generic parts in this example make the code reusable across many types. The first sign that it's generic is the T type parameter. You declare type parameters in square brackets before the function arguments. Here, T represents the element type used by both lists.

Mojo favors names that show purpose over type. It adopts naming conventions shared across modern programming languages, including Rust and C++, which helps keep generic signatures and parameter names readable.

Mojo type parameter names use PascalCase. They may be short (T, E) or descriptive (ErrorType, Element, Count). Common conventions include:

    T, U, V general type parameters
    K, V key and value types
    H hasher types or functions
    E error types, as in Result<T, E>

When you call all_equal(), the compiler looks at the call site to determine T's type:

print("Int (Expect True):\t",
    all_equal([1, 2, 3], [1, 2, 3])) # True
print("Int (Expect False):\t",
    all_equal([1, 2, 3], [4, 5, 6])) # False
print("String (Expect True):\t",
    all_equal(["hello", "world"], ["hello", "world"])) # True
print("String (Expect False):\t",
    all_equal(["hello", "world"], ["goodbye", "world"])) # False

The compiler generates a concrete function, a custom, type-specific version of all_equal() for each type used. In this example, it creates two versions, one for Int, and one for String.

In type theory terms, the parametric all_equal() function is polymorphic. The concrete, type-specific versions are monomorphic.
Type requirements

Equatable is required by the example algorithm. Copyable is often needed for loops and container elements. These are compiler-enforced, so you won't have to guess which ones you need.

Keep your requirements as minimal as your code allows:

T: Equatable & Copyable

The & symbol you see here forms your trait composition. It means: "both of these requirements apply."

Always use the fewest bounds you can get away with. This offers your code the greatest flexibility and the widest type compatibility. For example, if you remove the Equatable trait bound from all_equal(), the code won't compile. Mojo can't guarantee that all possible values of T implement support for the != operator.

A type provides operator support for != by implementing the __ne__() dunder function.

The Mojo compiler reliably emits errors when your code calls a function or accesses a member that isn't enumerated by trait bounds. The error message shows that the generic type is underspecified with respect to behavior: the Mojo compiler can't assume the requested functionality exists for every type that satisfies the current bounds, so it reports that back to you.

To fix this, you expand the generic type's trait bounds, increasing its vocabulary. You normally do this with trait composition, adding additional traits to the type parameter's bounds. Although these additional bounds are technically restrictions or constraints, they actually increase the API surface available to the generic function. Adding more traits exposes functionality that generic abstraction hides, allowing the compiler to reason more precisely about lifetimes, side effects, and which type members are visible.

To conclude, generic errors occur when a type's trait bounds don't include the functionality the code relies on. Fixing them means you need to expand those bounds to make that behavior visible to the compiler.

In type theory terms, bounds ensure your code is sound. This means that trait elements used by your implementation are guaranteed to exist for any type that satisfies the bounds.
Generic parameter types

In this example, both parameters are Lists that use the same element type:

lhs: List[T], rhs: List[T]

Using T for both arguments ensures your loop compares like-with-like and prevents type mismatches.

The previous example showed lists containing elements of type T. Generic types can also be used without embedding them into container types, like you see here:

def my_generic_fn[T: AnyType](value: T):

This function accepts any type, because its only limit is AnyType. AnyType is the root of the trait hierarchy. It's the baseline that all types extend. Using AnyType means the value has no guaranteed destructor or lifetime management. Outside of reflection (type introspection), this function is effectively useless.
Generic types

So far, you've seen generics applied to methods and functions. Mojo also lets you define generic types. These use compile-time parameters to define both their fields and their methods.

Generic types let you package a reusable shape: stored fields plus supported operations. Instead of coding PairInt, PairString, and so on, you write one Pair[T] and let the compiler generate specialized versions for each concrete T you use.

comptime ComparableValue = Equatable & ImplicitlyCopyable

@fieldwise_init
struct Pair[T: ComparableValue](ComparableValue):
    var left: Self.T
    var right: Self.T

    def __eq__(self, other: Pair[Self.T]) -> Bool:
        return self.left == other.left and
               self.right == other.right

Like generic functions, generic types use placeholder parameters. The difference is that a generic type uses those parameters in its storage as well as in its methods. Here, Pair stores two values of the same element type and implements equality by comparing its fields.

Traits are what make generic types practical. Those bounds describe the requirements the type definition depends on. In this example, Pair needs to compare values, so T must be equatable. It also needs to assign and copy values in common operations, so it applies a trait composition bound to T:

comptime ComparableValue = Equatable & ImplicitlyCopyable

The Pair definition applies this bound in two places: to the type itself (in parentheses) and to the type parameter (in square brackets), which is used to define its fields.

struct Pair[T: ComparableValue](ComparableValue):

    For the parentheses after the type, you can say the struct "conforms to" or "implements" this bound. This is a promise. The Pair type is saying, "I am a ComparableValue."
    For the square brackets, the trait bound restricts which values callers can pass. This is a requirement: "Any value used with Pair must have type T, and T must be a ComparableValue."

This makes Pair[T] usable anywhere a ComparableValue is required (because of the parentheses) and ensures all T values conform (because of the square brackets).

So far, these examples use type parameters. Generic types can also take non-type parameters, which you'll see next.
Adding non-generic parameters

Generic code doesn't keep you from using other parameters. Include whatever parameters you need to fully define the functionality you're building:

struct ExampleStruct:
    def example[
        T: Writable & Copyable,   # generic type parameter
        count: Int,               # parameter
    ](
        self,
        data: String,      # function argument
        init_value: T      # generic argument
    ) -> String:

By convention, Mojo uses lower_snake_case for parameter names that define compile-time values. This helps distinguish them from type parameters.
Downcasting safely: conforms_to + trait_downcast()

Sometimes a generic parameter's bound is broader than what you actually need. You may require operations from a specific trait that a parameter's bound doesn't guarantee. In those cases, Mojo provides a guarded downcast pattern using conforms_to() and trait_downcast().

Downcasting is valid when the value's underlying type actually implements the target trait. Check for that conformance first. Once it's confirmed, you can safely downcast and treat the value as having that trait.

This pattern keeps your code correct and makes fallback behavior explicit:

def process[T: AnyType](value: T):
    comptime if conforms_to(T, Writable & ImplicitlyCopyable):
        var w = trait_downcast[Writable & ImplicitlyCopyable](value)
        print(w)
    else:
        print("<not writable>")
        # Alternatively, you can fail at compile time using assertions
        # with `comptime assert`

Key points:

    conforms_to(T, Writable & ImplicitlyCopyable) checks whether type T conforms to the given trait composition.
    trait_downcast() rebinds a value to a trait-bounded view at compile time.

Only call trait_downcast() inside the guarded branch. Use the else branch to handle non-conforming values. If a trait is always required, apply the bound directly, such as T: Writable, instead of using a guarded downcast.

Limit trait_downcast() to cases where you can't express the requirement as a generic bound. This pattern most often appears in reflection code, where you need to handle values conditionally based on their traits. See reflection for details.
When generics get more complex

The examples on this page focus on the most common and useful cases for working with generics. As you progress to larger APIs and more abstract code, you'll likely encounter a few more generic patterns. Here's a quick summary of the most likely ones you'll see:

    Associated (dependent) types: Some traits declare associated types. For example, collection and iteration traits use an associated element type (commonly named Element) to specify the values the collection owns. Associated types can also constrain behavior. For instance, they can require that collection elements be hashable or implicitly destructible.

    Generic methods in non-generic types: Types don't have to be generic themselves to use generics. Individual methods can introduce their own generic type parameters when only part of an API needs them. This is common when a type processes values of different types, but isn't tied to a single type at construction.

    Multiple type parameters: Most generics use a single type parameter, but some algorithms work with two or more. In those cases, traits describe how the parameters relate to each other, not just what each type can do on its own. Common examples include dictionaries and hash tables (keys and values), caches and memoization (also keys and values), database relations (source and target types), operation results (values and errors), and zippers that operate across multiple types.

These topics build on the same core ideas shown here: type parameters, bounds, and specialization. Bounds can appear on functions, methods, types, or helper aliases using comptime, keeping capability requirements close to where they matter. You won't need these patterns for simple cases, but they become important as generic code grows.
Generics and explicit destruction

Explicitly destroyed types sometimes don't work well with some generic code. The problem with this combination isn't generics, it's lifetime management.

Explicit destruction exists so you can control teardown. With it, you can create destructors that take arguments, follow multiple destruction paths, or allow chaining and error-raising.

Explicitly destroyed types provide a safe and flexible way to manage value lifetimes. Generic code that copies or moves values can't see or honor that logic. Once a value is copied or transferred, you've lost control of how (or whether) it's cleaned up, and your code won't compile.

The danger zone includes:

    Generic code that manages lifetimes
    Generic code that copies or transfers values

These scenarios occur most often with generic containers, collections, and iterators, which copy or take ownership of their members and decide when transfer and destruction happen.

The safe zone includes:

    Generic operations that don't touch lifetimes

Think comparisons, predicates, and pure operations.

As a working rule of thumb, if your generic code needs to own, copy, or decide when a value dies, avoid explicitly destroyed types. Add ImplicitlyDestructible bounds to bypass any issues.
Conditional trait conformance

Conditional trait conformance uses checks before allowing a type to adopt a trait. If the condition is satisfied, the type conforms. It must fulfill the trait's requirements by providing required methods and associated types, and it gains any default implementation provided by the trait. The type can then be used anywhere the trait is required, such as when passing it to a function that expects a conforming type.

When the condition isn't satisfied, Mojo skips the conformance. You can't use the fully concrete type in contexts that require the trait.
Example: derived conformance

In the following declaration, Mojo conforms Wrapper to Writable when its parameter, T, is also Writable:

comptime BaseTraits = Copyable & ImplicitlyDestructible
@fieldwise_init
struct Wrapper[T: BaseTraits](
    Writable where conforms_to(T, Writable)
):
    var value: Self.T

When conforming to Writable, Wrapper doesn't need to implement any methods to enable printing. The trait provides a default implementation of the write_to() method.

Now consider a type that isn't Writable, that you might add into a Wrapper:

@fieldwise_init
struct NotWritable(BaseTraits):
    var data: Int

When instantiated with an Int or String (both Writable), the Wrapper gains Writable conformance. With NotWritable, you can build a struct, but you can't print it:

var w_int = Wrapper[Int](42)  # Int is Writable
print(w_int)  # Wrapper[Int](value=42)


var w_str = Wrapper[String]("Hello")  # String is Writable
print(w_str)  # Wrapper[String](value=Hello)


# OK: only `Writable` conformance is unavailable
var w_not_writable = Wrapper[NotWritable](NotWritable(10))
# print(w_not_writable)  # Compile-time error:
# invalid call to 'print': could not convert element of 'values' with
# type 'Wrapper[NotWritable]' to expected type 'Writable'

Since NotWritable doesn't conform to Writable, the conditional conformance check fails and Wrapper[NotWritable] doesn't adopt the Writable conformance. In contrast, Int and String are Writable and their wrappers gain the conformance. This allows their instances to print by using the write_to() method provided by Writable.

This pattern is standard for single-type containers like Optional[T], Box[T], Lazy[T], and List[T]. It says: "This type can do X if its inner type can do X."
Example: parts conformance

Conditional conformance has another standard pattern for types with multiple distinct components: Result[T, E], Pair[L, R], Dict[K, V], and similar. It says: "This type can do X if each of its parts can do X":

comptime BaseTraits = Copyable & ImplicitlyDestructible

@fieldwise_init
struct Pair[L: BaseTraits, R: BaseTraits](
    Hashable where conforms_to(L, Hashable) and conforms_to(R, Hashable)
):
    var left: Self.L
    var right: Self.R


@fieldwise_init
struct NotHashable(BaseTraits):
    var data: Int

In this example, Pair uses two distinct type parameters. When both are Hashable, the concrete Pair type becomes Hashable, gaining the hash() method from the trait:

var pair = Pair[Int, String](left=1, right="one")
var hash = hash(pair)
print(hash)  # Prints the hash of the pair

# OK: only hashing is unavailable
var pair2 = Pair[Int, NotHashable](left=1, right=NotHashable(10)) # Constructible
# var hash2 = hash(pair2) # Compile time error

Example: conditional method access

When a type adopts a trait, it must satisfy the trait's required methods. The same conditions that determine whether a type can conditionally conform to a trait are usually the same conditions used to gate required method implementations.

You express these method tests with where clauses. If true, the method is available. If not, the method can't be used for that specialized type. The following example walks you through this process and how it works.

You can use Boolable instances in if statements. You make your Wrapper testable by conforming to the Boolable trait. This trait requires a new method, __bool__():

@fieldwise_init
struct Wrapper[T: BaseTraits](
    Writable where conforms_to(T, Writable),
    Boolable where conforms_to(T, Boolable), # New conditional conformance
    ):
    var value: Self.T

    # New method
    def __bool__(self) -> Bool where conforms_to(Self.T, Boolable):
        return trait_downcast[Boolable](self.value).__bool__()

In this example, the condition on __bool__() is the same as the one used for the Boolable conditional conformance. The where clause gates the method. If the condition is true, the method is available. If not, it isn't.

Since the method and the conformance use the same condition, they stay aligned. You won't end up in a situation where the wrapped type conforms but the method isn't available, or where the method exists without the corresponding conformance.

You see this in the following examples:

var w_str = Wrapper[String]("Hello")
if w_str:  # Chooses the non-empty branch
    print(t"Non-empty string \"{w_str.value}\" is truthy")
else:
    print(t"Empty string \"{w_str.value}\" is falsy")

var w_empty_str = Wrapper[String]("")
if w_empty_str:  # Chooses the empty branch
    print(t"Non-empty string \"{w_empty_str.value}\" is truthy")
else:
    print(t"Empty string \"{w_empty_str.value}\" is falsy")

The NotWritable type from earlier in this section has no traits beyond the base Copyable and ImplicitlyDestructible. Therefore it isn't Boolable and the condition on the __bool__() method will fail:

@fieldwise_init
struct NotWritable(BaseTraits):
    var data: Int

var w_not_writable = Wrapper[NotWritable](NotWritable(10))

# The instance can't be used as a boolean value due to the
# conditional method access for `__bool__()`.

# Compiler error: the method condition is false
if w_not_writable:
    print(t"NotWritable with data {w_not_writable.value.data} is truthy")
else:
    print(t"NotWritable with data {w_not_writable.value.data} is falsy")

This condition on your method isn't conditional trait conformance. It's conditional method access. In this example, the method and the conformance use the same condition, so they stay aligned.

That said, where clauses on methods and functions are useful even when you're not working with trait conformance. For example, you might gate a method so it only works with non-empty lists, real numbers, or with values that fall within the index bounds of a collection.
Example: strategy delegation

The previous examples propagate conformance from a data parameter — "this container can do X if its elements can." There is a second, equally important pattern: propagating conformance from a strategy or policy parameter.

Consider a struct parameterized on a behavior strategy. Some strategies have a special property (expressed as a sub-trait), and you want the struct to inherit that property only when its strategy has it. This lets generic code query the struct's conformance directly, rather than threading the strategy type through every callback and function signature.

trait ShardStrategy:
    @staticmethod
    def shard_rows(r: Int, tp: Int) -> Int: ...
    @staticmethod
    def shard_cols(c: Int, tp: Int) -> Int: ...

trait NodeLocal(ShardStrategy):
    ...

struct RowShard(ShardStrategy):
    @staticmethod
    def shard_rows(r: Int, tp: Int) -> Int: return r // tp
    @staticmethod
    def shard_cols(c: Int, tp: Int) -> Int: return c

struct PrincipleNodeLocal(NodeLocal):
    @staticmethod
    def shard_rows(r: Int, tp: Int) -> Int: return r
    @staticmethod
    def shard_cols(c: Int, tp: Int) -> Int: return c

struct PlacedSlot[S: ShardStrategy, name: StringLiteral](
    ShardStrategy,
    NodeLocal where conforms_to(S, NodeLocal),
):
    comptime NAME: StaticString = Self.name

    @staticmethod
    def shard_rows(r: Int, tp: Int) -> Int: return Self.S.shard_rows(r, tp)
    @staticmethod
    def shard_cols(c: Int, tp: Int) -> Int: return Self.S.shard_cols(c, tp)

PlacedSlot always conforms to ShardStrategy, but it only conforms to NodeLocal when its strategy parameter S does. The compiler evaluates this per-instantiation:

comptime DISTRIBUTED = PlacedSlot[RowShard, "q_proj"]
comptime LOCAL = PlacedSlot[PrincipleNodeLocal, "norm"]

comptime if conforms_to(DISTRIBUTED, NodeLocal):
    ...  # not taken — RowShard is not NodeLocal

comptime if conforms_to(LOCAL, NodeLocal):
    ...  # taken — PrincipleNodeLocal is NodeLocal

The key benefit is eliminating plumbing parameters. Without conditional conformance, generic iteration over both distributed and node-local slots requires passing the strategy type as a separate parameter through every callback:

# Before: two type parameters, S threaded everywhere
trait WeightIterable:
    @staticmethod
    def for_each_weight[
        func: def[S: ShardStrategy, T: Named](String) capturing -> None,
    ](): ...

# Call sites must redundantly pass the strategy
func[RowShard, Self.Q_PROJ]("q_proj")
func[PrincipleNodeLocal, Self.EMBED]("embed")

With conditional conformance, the NodeLocal-ness is baked into the PlacedSlot type itself. Callbacks need only one type parameter and query it directly:

# After: single type parameter, no plumbing
trait WeightIterable:
    @staticmethod
    def for_each_weight[
        func: def[T: Named](String) capturing -> None,
    ](): ...

# Call sites are simpler
func[Self.Q_PROJ]("q_proj")
func[Self.EMBED]("embed")

# Consumer branches on T directly
@parameter
def collect[T: Named](name: String):
    comptime if conforms_to(T, NodeLocal):
        node_local_weights.append(name)
    else:
        distributed_weights.append(name)

This pattern applies whenever a struct delegates behavior to a strategy parameter and consumers need to branch on the strategy's properties: placement policies, memory allocation strategies, serialization backends, scheduling disciplines, and similar.
Conditional trait composition

Mojo supports flexible condition composition, as shown in these examples:

    Unconditional: No special clauses.

    struct Foo(Copyable, ImplicitlyDestructible):

    Simple condition: as shown in Wrapper.

    struct Wrapper[T: BaseTraits](
        Writable where conforms_to(T, Writable)
    ):

    Hybrid: Mixes unconditional and conditional traits.

    struct Foo[T: AnyType](
        Copyable, Writable where conforms_to(T, Writable)
    )

    Multiple aligned conditions: as shown in Pair.

    struct Pair[L: BaseTraits, R: BaseTraits](
        Hashable where conforms_to(L, Hashable) and conforms_to(R, Hashable)
    ):

    Multiple independent conditions:

    struct Foo[T: AnyType](
      Writable where conforms_to(T, Writable),
      Hashable where conforms_to(T, Hashable),
    )

Comptime constraints and assertions

Mojo's constraint system lets you express program guarantees that go beyond what the type system provides. This page explains how to use this system effectively.

A constraint, defined with the where keyword, represents a precondition for calling a function or instantiating a struct:

def fib[x: Int where x >= 0]() -> Int:
    ...

Here the fib() function requires its x parameter to be greater than or equal to 0. The expression x >= 0 is called a proposition. With the exception of very simple expressions, Mojo doesn't evaluate these propositions literally. Instead, it analyzes them symbolically, tracking a list of propositions that are known to be true in the current scope. Understanding this system of symbolic propositions is key to using constraints effectively.

Mojo also supports compile-time assertions, which test a proposition at compile time—if the assertion evaluates to false, compilation fails:

comptime assert x >= 0, "x must be greater than or equal to 0."

Defining constraints

You can use constraints in the following contexts:

    In the parameter list of a function or struct, to constrain the values that you can bind to one or more parameters.

    def fib[x: Int where x >= 0]() -> Int:

    A constraint can involve multiple parameters, as long as the where clause comes after the parameters:

    def subspan[start: Int, end: Int where end > start](self) -> Self:

    In a method declaration, to constrain the availability of the method based on parameters of the parent struct.

    def sort() where conforms_to(Self.T, Comparable):

    In the trait conformance list for a struct, to declare that a struct conforms to a trait only when certain conditions are met.

    struct MyContainer[T: AnyType](Copyable where conforms_to(T, Copyable)):

    You'll use this form, called conditional trait conformance, when defining generic types. For details, see the section on conditional trait conformances.

Symbolic propositions

The constraint system works with symbolic propositions.

def first[size: Int where size > 0](array: InlineArray[_, size]) -> array.T:
  ...

In this case, size > 0 is a proposition that the constraint system tracks.

By analogy, when you annotate a type on an argument you ask the compiler to ensure callers only pass that type. When you annotate a constraint on a function, you ask the compiler to ensure that the proposition is true at the call site.

But the compiler can't test every proposition at every call site without running or interpreting unbounded amounts of code, which would vastly expand compile times. Instead, callers need to explicitly introduce "knowledge" into the constraint system.
Introducing knowledge

The compiler tracks knowledge by scopes: each scope contains a set of known true propositions, also known as "knowledge," and nested scopes accumulate knowledge from outer scopes.

There are four ways to introduce knowledge to the system:

    Inside a struct declaration, all constraints declared on the struct are known, because concrete instances of the struct type can only be created when the proposition is true:

    struct List[size: Int where size >= 0]:
      # Knowledge base:
      # - `size >= 0`

    Inside a function, all constraints declared on the function are known, because callers can only call the function if they guarantee the constraint holds:

    def create_list[size: Int where size >= 0]() -> List[size]:
      # Knowledge base:
      # - `size >= 0`

    def create_list[size: Int]() -> List[size] where size >= 0:
      # Knowledge base:
      # - `size >= 0`

    Inside a comptime if, the if condition is known, because the code within the body is only instantiated if the condition is true:

    comptime if size >= 0:
      # Knowledge base:
      # - `size >= 0`

      comptime if size.is_even():
        # Knowledge base:
        # - `size >= 0`
        # - `size.is_even()`

    After a comptime assert, the asserted condition is known, because the compiler won't instantiate a function if a comptime assert condition is false. So any code after it is only instantiated if the assertion didn't fire:

    comptime assert size >= 0
    # Knowledge base:
    # - `size >= 0`

    comptime assert size.is_even()
    # Knowledge base:
    # - `size >= 0`
    # - `size.is_even()`

Satisfying constraints with knowledge

Any time you call a function that has constraints, the system inspects the set of known-true propositions at the call site and determines whether the known set of propositions is a superset of the required set of propositions declared on the callee.

The constraint system treats all propositions symbolically: it doesn't know or care what the expression means and doesn't interpret the code. With a few very simple exceptions (discussed later), the system doesn't perform symbolic math on your behalf. Instead, it treats these propositions as opaque and only uses knowledge you've explicitly introduced in the calling scope.

# We have a function that wants a non-empty list.
def print_first[size: Int where size >= 1](l: InlineArray[Int, size]):
    ...


# A wrapper that wants a size >= 2 list.
def print_first_two[size: Int where size >= 2](l: InlineArray[Int, size]):
     # Error: invalid call to 'print_first': lacking evidence to prove
     # correctness
    print_first[size](l)
    # ...

Mojo checks constraints without knowing all the call sites, and these expressions can reference symbolic parameter values-for example is_prime(x) where x is unknown. Because of this, Mojo can't interpret is_prime() for all possible values of x, nor can it make logical deductions. For example if is_prime(x) and x > 2 are true, it can't deduce that is_odd(x) must also be true.
Compile-time assertions

Use comptime assert to introduce a known-true proposition at a specific point in the code:

comptime assert x > 0, "x must be greater than 0."

The message is optional. If the condition evaluates to false at compile time, compilation fails and the compiler shows the message (or a generic message if none is specified).

Mojo adds the asserted condition to the list of "known true" propositions for any code following the assertion.

Constraints and assertions serve complementary roles: a where clause exposes a requirement that callers must prove, and comptime assert is one way to satisfy that requirement.
Limited evaluation of propositions

In general, it's safe to assume that the constraint system doesn't evaluate propositions directly, and that all knowledge needs to be provided explicitly. However, the constraint system does apply a very limited amount of "smartness" for very common cases. These are provided as a convenience and should not be treated as the norm.

The goal is to provide a consistent, predictable experience so users can recognize these patterns as they become more familiar with the system.
Simple implication

A known proposition of the form A and B can satisfy a requirement of A by itself (or B by itself), even though symbolically they aren't identical propositions.

def create_even_list[
  size: Int where size >= 0 and size.is_even()
]() -> List[size]:
  # No need to individually assert that `size >= 0`.
  return create_list[size]

Similarly, A implies A or B for any B.
Canonicalization

Sometimes there is more than one way to write the same expression. The system always simplifies expressions into a normal form internally, so expressions that aren't identical on the surface may be seen as identical by the system.

The following are self-explanatory (assume x: Int):

    x > 0 == x >= 1
    x >= 2 == not (x < 2)
    x + x == 2 * x

There's a special case for function calls. When invoking functions as part of a proposition, the system treats the entire function call as opaque. Only two identical function calls are the same.

def is_even(x: Int) -> Bool:
  return x % 2 == 0

def needs_even[x: Int where is_even(x)]():
  pass

def forward_even_bad[x: Int where x % 2 == 0]():
  # ERROR: Needs evidence for `is_even(x)`.
  needs_even[x]

def forward_even_good[x: Int where is_even(x)]():
  # SUCCESS
  needs_even[x]

Builtin functions

Some functions in the standard library can be evaluated in where clauses. These functions implement simple operations on core types (for example, Int numerics), which are known to the compiler. These functions and can be inlined into the calling expression when called in a parameter context.

However, there's no way to identify these builtin functions without looking at the source code, and whether a given function is builtin may change without notice.

If you need a predicate that works transparently rather than opaquely, consider implementing it as a parametric comptime value instead, which is always inlined.

comptime is_even[x: Int]: Bool = x % 2 == 0

Context-free folding

This is more of an extreme case of canonicalization than a separate category, but may catch people by surprise.

Certain primitive operations on constants can be canonicalized into a new constant. For example:

    1 + 1 == 2
    4 % 2 == 0
    1 == 1 == True

This extends to expressions with a mix of constants and non-constants:

    1 + x + 1 == 2 + x

Concretely, this means you may omit explicitly providing knowledge for propositions that are simple operations on constants:

def create_my_list() -> List[2]:
  # No need to provide evidence for `2 >= 0`.
  return create_list[2]()

Best practices

The examples here focus on functions (declaring constraints when writing functions and satisfying constraints when calling functions), but the tips generalize to other forms of constraints such as struct parameter constraints and conditional trait conformances.
Writing functions

When writing a function, how do you decide whether a where clause is right for you?

Figure 1. Deciding when to use constraints
Q1: Does your function handle the entire domain of its input types?

If you can write your function so that it returns a result or throws an error on all inputs, you don't need to care about constraints at all. Just write your function as usual.

For example, the following functions do not need constraints since they're guaranteed to return or throw:

def create_list_opt[size: Int]() -> Optional[List[size]]:
  comptime if size < 0:
    return None
  return ...

def create_list_raise[size: Int]() raises -> List[size]:
  comptime if size < 0:
    raise Error("negative size!")
  return ...

Constraints are only for cases where you don't want to check for exceptional cases at execution time.

    Constraints ask the type checker to rule out exceptional cases so that a type-checked program guarantees you (as the function author) don't need to handle these cases in your function body using run-time resources.
    As always, enforcing static guarantees isn't free. You're trading off run-time error checking logic for compile-time proof writing. In the absence of hard limits that prevent you from error checking at run time, it comes down to your preference for user experience.

Q2: Is this limitation a central concept of your code?

If the condition represents a concept that is central to your code, a dedicated type may be easier than sprinkling where everywhere.

Examples:

    A SIMD library that needs to represent SIMD width values, which aren't arbitrary integers.
    A filesystem library that needs to represent paths that follow specific format rules.
    A network library that needs to represent port numbers, which must be in the valid range.

This is a good approach because the constraint is proven once (at construction) and the refined value can be passed around unconstrained until it needs to be disassembled. Your APIs stay simpler: fewer where clauses, fewer repeated asserts.

A new type doesn't come for free though, as it's no longer freely interoperable with the original type. Make sure that the semantics are distinct enough to warrant the extra code. For example, writing a new type usually means implementing dunder methods corresponding to common operators (addition, subtraction, equality, and so on), which preserve the semantics of the new type.
Q3: Is the constraint understandable and provable by the caller?

If yes, use constraints.

Make the condition part of the function's contract so that callers must prove it.

This tends to be the right choice when:

    The condition is a user-facing requirement that the user can understand.
    Callers typically already need to handle the "bad" case themselves (for example, they may already have a comptime if that branches on this condition).

Example:

def take_prefix[n: Int, len: Int where 0 <= n <= len](...) -> ...:
  ...

One practical heuristic: if a user can read the function signature and immediately understand what they did wrong when the constraint fails, where is likely the right tool.

If the constraint is not understandable or not provable by the user, use assert or abort.

If the "bad case" isn't something users would understand, adding a where constraint only causes more confusion, as users can't reasonably prove it themselves.

    This typically happens when your library returns a value that has some internal constraints on it and expects users to pass it back with those constraints. For example, a communication library passes a device handle to the user as an Int, which has properties that are only known internally (for example, non-negative, special bits). Library APIs accepting this handle can't expect the user to prove these.
    There are usually more type-safe ways to achieve the same thing, so only do this if you don't care about type safety. For example, a more type-safe approach is to introduce a new type for the device handle so users are less likely to accidentally pass another Int or modify the returned value unexpectedly.

Summary

    Use a dedicated type when the condition is a common refinement that should be proven once and reused everywhere.
    Use where when the condition is a user-understandable precondition.
    Use comptime assert or abort for internal inconsistencies in your library where a user can't do anything with the failure.

Calling functions

When calling a function, how do you show proof that you've satisfied the constraint?

This is where the constraint system actively guides you into handling exceptional cases.

Figure 2. Using constrained APIs
Q1: Do the constrained parameters come from a parent parameter list?

If the constrained parameter you're passing is from the function's parameter list, or is a parameter on the enclosing struct, you're basically "forwarding" a value from one parameter list to another. Go to Q2.

If not, you computed the value inside the function body, and you're in "construction" territory. Go to Q3.
Q2: Does the same limitation apply to your function?

If you're forwarding a constrained parameter into another constrained API, it's usually a hint to propagate the same requirement onto your own function.

For example, to propagate the constraint to your callers:

def create_list[size: Int where size >= 0]() -> List[size]:
  ...

# This function is also only meaningful when size >= 0.
# Make that part of the contract too.
def create_list_and_fill[size: Int where size >= 0]():
  comptime xs = create_list[size]()
  ...

But if you want your function to accept a wider input domain, your job is to handle both cases explicitly using a comptime if.

For example, narrow the domain by checking before the call:

def create_list_and_fill[size: Int]() -> Optional[List[size]]:
  comptime if size >= 0:
    return needs_nonneg[size]()
  else:
    return None

Q3: Do you know that the parameter already satisfies this condition?

If the parameter didn't come from a parent parameter list, it must have been computed in the body.

Decide whether the desired constraint holds by construction.

If it does, indicate this to the constraint system via a comptime assert. Note that this is you explicitly taking over the burden of proof.

Don't be afraid to write these asserts. The symbolic nature of the constraint system means that it is conservative—logical deductions that are obvious to you aren't always "obvious" to it. Adding comptime assert is not a code smell, but rather an inseparable part of working within the constraint system.

For example, given a computed parameter that is always valid:

# The constraint on hi & lo guarantees a valid size.
# Introduce this piece of knowledge explicitly.

  comptime size = hi - lo
  comptime assert size >= 0, "span is guaranteed non-negative"
  return create_list[size]()

But if the constraint does not necessarily hold, insert a comptime if and handle both cases explicitly.

For example, a computed parameter that may be invalid:

# This version does NOT have a constraint on its inputs.
# Branch on the computation to handle both cases.
def create_list_from_span[lo: Int, hi: Int]() -> Optional[List[hi - lo]]:
  comptime size = hi - lo
  comptime if size >= 0:
    return create_list[size]()
  else:
    return None

Summary

    Constraints are part of the API contract.
        A where clause is a precondition that the API author requires the caller to prove.
    The compiler doesn't automatically derive evidence for callers.
        The caller must explicitly provide evidence that the precondition is always satisfied.
    If a caller gets a "lacking evidence" error, they can:
        Add a constraint to push the requirement onto their own callers.
        Branch on the condition (comptime if).
        Assert an invariant (comptime assert).

Materializing compile-time values at run time

Mojo's compile-time metaprogramming makes it easy to make calculations at compile time for later use. The process of making a compile-time value available at run time is called materialization. For types that can be trivially copied, this isn't an issue. The compiler can simply insert the value into the compiled program wherever it's needed.

comptime threshold: Int = some_calculation()  # calculate at compile time

for i in range(1000):
    my_function(i, threshold)  # use value at runtime

However, Mojo also allows you to create instances of much more complex types at compile-time: types that dynamically allocate memory, like List and Dict. Re-using these values at run time presents some questions, like where the memory is allocated, who owns the values, and when the values are destroyed.

This page describes when Mojo materializes values, and presents some techniques for avoiding unnecessary materialization of complex values.
Implicit and explicit materialization

When you use a comptime value at run time, you're explicitly or implicitly copying the value into a run-time variable:

comptime comptime_value = 1000
var runtime_value = comptime_value

This process of moving a compile-time value to a run-time variable is called materialization. If the value is implicitly copyable, like an Int or Bool, Mojo treats it as implicitly materializable as well.

But types that aren't implicitly copyable present other challenges. Consider the following code:

def lookup_fn(count: Int):
    comptime list_of_values = [1, 3, 5, 7]

    for i in range(count):
        # Some computation, doesn't matter what it is.
        idx = dynamic_function(i)

        # Look up another value
        lookup = list_of_values[idx]

        # Use the value
        process(lookup)

This looks reasonable, but compiling it produces an error on this line:

lookup = list_of_values[idx]

cannot materialize comptime value of type 'List[Int]' to runtime
because it is not 'ImplicitlyCopyable'

Just like Mojo forces you to explicitly copy a value that's expensive to copy, it forces you to explicitly materialize values that are expensive to materialize, by calling the materialize() function.

Here's the code above with explicit materialization added:

def lookup_fn(count: Int):
    comptime list_of_values = [1, 3, 5, 7]

    for i in range(count):
        idx = dynamic_function(i)

        # This is the problem
        var tmp: List[Int] = materialize[list_of_values]()
        lookup = tmp[idx]
        # tmp is destroyed here

        process(lookup)

This code materializes the list of values inside of the loop, which includes dynamically allocating heap memory and storing the four elements into that memory. Because the last use of tmp is on the next line, the memory then gets deallocated before the loop iterates. This creates and destroys the list on every iteration of the loop, which is clearly wasteful.

A more efficient version would materialize the list outside of the loop:

def lookup_fn(count: Int):
    comptime list_of_values = [1, 3, 5, 7]

    var list = materialize[list_of_values]()
    for i in range(count):
        idx = dynamic_function(i)

        lookup = list[idx]

        process(lookup)
    # materialized list is destroyed here

This is why Mojo requires you to explicitly materialize non-trivial values; it puts you in control of when your program allocates resources.
Global lookup tables

Mojo doesn't currently have a general-purpose mechanism for creating global static data. This is a problem for some performance-sensitive code where you want to use a static lookup table. Even if you declare the table as a comptime value, you need to materialize it each time you want to use the data.

The global_constant() function provides a solution for storing a compile-time value into static global storage, so you can access it without repeatedly materializing the value. However, this currently only works for self-contained values which don't include pointers to other locations in memory. That rules out using collection types like List and Dict.

The easiest way to use global_constant() is with InlineArray, which allocates a statically sized array of elements on the stack. The following code uses global_constant() to create a static lookup table.

from std.builtin.globals import global_constant

def use_lookup(idx: Int) -> Int64:
    comptime numbers: InlineArray[Int64, 10] = [
        1, 3, 14, 34, 63, 101, 148, 204, 269, 343
    ]
    ref lookup_table = global_constant[numbers]()
    if idx >= len(lookup_table):
          return 0
    return lookup_table[idx]

def main() raises:
    print(use_lookup(3))

At compile time, Mojo allocates the numbers array, and then the global_constant() function copies it into static constant memory, where the code can reference it without requiring any dynamic logic to create or populate the array. At run time, the lookup_table identifier receives an immutable reference to this memory.

Note the use of ref lookup_table to bind the reference returned by global_constant(). Using var lookup_table would cause a compiler error, because it would trigger a copy, and InlineArray doesn't support implicit copying.
Using the comptime keyword

Another approach that you can use to avoid materializing a complex value is to use the comptime keyword to control when Mojo evaluates an expression. Assigning an expression to a comptime value causes Mojo to evaluate the expression at compile time.

For example, if you want to force a function to run at compile time:

comptime tmp = calculate_something()  # executed at compile time
var y = x * tmp  # executed at run time

If you're only creating a comptime value for a single use, you can use a comptime sub-expression instead:

var y = x * comptime (calculate_something())

This works exactly like the previous example, without creating a named temporary value. The comptime keyword here tells Mojo to evaluate the expression inside the parentheses (calculate_something()) at compile time.

For example, you can use a comptime sub-expression when working with the Layout type, which determines how you store and retrieve data in a LayoutTensor. Materializing a Layout requires dynamic allocation, which isn't supported on GPUs. So calling this code on a GPU produces an error:

comptime layout = Layout.row_major(16, 8)
var x = layout.size() // WARP_SIZE  # Can't implicitly materialize layout

A comptime sub-expression fixes this issue:

comptime layout = Layout.row_major(16, 8)
var x = comptime (layout.size()) // WARP_SIZE

Now, the expression layout.size() gets evaluated at compile time, so there's no need to materialize the layout.

You could also achieve the same effect using a named comptime value.

comptime layout = Layout.row_major(16, 8)
comptime layout_size = layout.size()
var x = layout_size // WARP_SIZE

The comptime sub-expression is just a more compact way to express the same thing.
Materializing literals

Literal values, like string literals and numeric literals are also materialized to their run-time equivalents, but this is mostly handled automatically by the compiler:

comptime str_literal = "Hello"  # at compile time, a StringLiteral
var str = str_literal  # at run time, a String.
var static_str: StaticString = str_literal  # or a StaticString

Both String and StaticString can be implicitly created from a StringLiteral, but without a type annotation, Mojo defaults to materializing StringLiteral as a String.

Reflection

Reflection helps you write code that inspects its own structure at compile time and reports information about types. This makes it possible to build features like structural validation, automatic comparisons, serialization, safer assertions, and richer error messages without hardcoding details for specific type implementations.
Caution

Mojo reflection is newly introduced and currently incomplete. Some reflection capabilities are limited, unstable, or not yet fully exposed through the language interface. This page describes the direction of the feature as well as the parts that are available today.

All examples reflect the state of the language at the time this page was published. They may change as reflection support matures.
Why reflection matters

Reflection is often used for serialization, such as converting values to JSON or MessagePack regardless of type. It also supports common structural tasks. Instead of writing equality, hashing, or copy logic by hand, reflection can apply those operations to all fields automatically.

In Mojo, reflection happens entirely at compile time. The compiler uses type information to generate specialized code, which avoids runtime cost while keeping the code flexible.

With this in mind, this page will introduce reflection through examples, so you can see working code to study and use. Reflection code is intentionally parameterized and compile-time heavy. The examples below show the patterns you can use in real reflection-based code.
Static vs. dynamic reflection

Mojo's reflection is static, meaning it runs at compile time. Mojo generates reflection information during compilation. It uses the type information in your code, including types, fields, and methods. This contrasts with dynamic reflection, which inspects program state at runtime.

Static reflection's timing affects what you can and cannot do. Reflection can only operate on types known to the compiler. If a variable's type depends on runtime conditions, reflection can't access it. However, since all reflection runs during compilation, there's no runtime overhead.

Mojo reflection operates on program structure, not live program state. Use it to serialize types, implement generic operations over struct fields (comparison, copying, arithmetic), and validate structural constraints at compile time.
Example: Present a type

This example demonstrates how to use reflection to inspect a type at compile time. It shows how to retrieve a type's name and iterate over a struct's fields, including their names and types. This pattern is the foundation for most reflection-based utilities, such as validation, copying, and equality checks.
Compile-time inputs

Mojo reflection APIs run at compile time and accept compile-time inputs.

Although reflection happens during compile time, it can drive behavior for runtime values. Knowing a value's type opens the door to retrieving type-specific metadata such as field counts, names, and types. That metadata can then be used to safely access and manipulate fields on concrete instances at runtime, as long as the access itself is guided by compile-time information.
Reflection calls

The reflection calls used in the following example include:

    get_type_name[](): Returns the name of a type.
    struct_field_count[](), struct_field_names[](), and struct_field_types[](): Return the count, names, and types of a struct's fields.

from std.reflection import (
    struct_field_count, struct_field_names,
    get_type_name, struct_field_types
)

def show_type[T: AnyType]():
    comptime type_name = get_type_name[T]()
    comptime field_count = struct_field_count[T]() # count of fields
    comptime field_names = struct_field_names[T]() # indexed list of field names
    comptime field_types = struct_field_types[T]() # indexed list of field types

    print("struct", type_name)

    comptime for idx in range(field_count):
        comptime field_name = field_names[idx]
        comptime field_type = get_type_name[field_types[idx]]()
        var intro = "├──" if idx < (field_count - 1) else "└──"
        print(intro, " var ", field_name, ": ", field_type, sep="")

@fieldwise_init
struct MyStruct:
    var x: String
    var y: Optional[Int]

comptime DefaultItemCount = 10

struct ParameterizedStruct[T: Copyable, item_count: Int = DefaultItemCount](Copyable):
    var list: List[Self.T]
    def __init__(out self):
       self.list = List[Self.T](capacity=Self.item_count)

def main():
    show_type[MyStruct]()
    show_type[Optional[Float64]]()
    show_type[Dict[Int, String]]()
    show_type[ParameterizedStruct[String, item_count=5]]()

Insights

    When working with reflection, you must use parameterized types in the square parameter brackets for calls that use reflection features, such as show_type[](). These elements can be concrete types like MyStruct or generic type parameters like T.
    This example uses a T: AnyType parameter for the show_type[]() function. This is the least restrictive constraint and allows the function to work with any type passed at the call site.
    Reflection functions are themselves parameterized. That means you do not call foo(). You call foo[T]() to reflect on a type.
    All processing of reflected information happens at compile time. For that reason, these reflection examples use comptime for and comptime if.
        In this example, comptime for allows the loop index to be used across iterations.
        Later examples show comptime if.

Using sep="" in show_type allows the colon to be flush left with the field name.
Program output

The following output shows the result of calling show_type[]() on three different types in main(). Each call prints the compiler's view of the type, including its fully resolved name and the structure of its fields:

struct reflect.MyStruct
├── var x: String
└── var y: std.collections.optional.Optional[Int]

struct std.collections.optional.Optional[SIMD[DType.float64, 1]]
└── var _value: std.utils.variant.Variant[<unprintable>]

struct std.collections.dict.Dict[Int, String, std.hashlib._ahash.
    AHasher[[0, 0, 0, 0] : SIMD[DType.uint64, 4]]]
├── var _len: Int
├── var _n_entries: Int
├── var _index: std.collections.dict._DictIndex
└── var _entries: List[std.collections.optional.Optional[std.collections.
    dict.DictEntry[Int, String, std.hashlib._ahash.
    AHasher[[0, 0, 0, 0] : SIMD[DType.uint64, 4]]]]]

struct show_type.ParameterizedStruct[String, 5]
└── var list: List[String]

When comparing the type names at the call sites with the names shown here, keep in mind that this output reflects how the compiler represents the type. This includes defaulted parameters, such as the dictionary's Hasher. Comptime type aliases are expanded, so Float64 displays as SIMD[DType.float64, 1].

This example showed how to read type information. The next examples show how to use that information to manipulate struct instances.
Example: Copying data

This example shows how reflected field information can drive real behavior. It uses reflection to copy data from one instance to another by iterating over fields and checking which ones are safe to copy. This pattern demonstrates how reflection enables generic, type-safe operations without hardcoding field access.

When a struct conforms to MakeCopyable, it gains the copy_to() method that uses reflection to perform the copy. Like all methods, you call it from an instance. For this method, you provide it with a target instance.

Its behavior is similar in spirit to ImplicitlyCopyable, but copy_to() limits copying to fields whose types conform to Copyable. It requires an already initialized target, avoiding matching values to __init__ arguments.

from std.reflection import struct_field_count, struct_field_types

trait MakeCopyable:
    def copy_to(self, mut other: Self):
        comptime field_count = struct_field_count[Self]()
        comptime field_types = struct_field_types[Self]()

        comptime for idx in range(field_count):
            comptime field_type = field_types[idx]

            # Guard: field type must be copyable
            comptime if not conforms_to(field_type, Copyable): continue

            # Perform the copy
            ref p_value = __struct_field_ref(idx, self)
            trait_downcast[Copyable & ImplicitlyDestructible](
                __struct_field_ref(idx, other)
            ) = trait_downcast[Copyable & ImplicitlyDestructible](
                p_value
            ).copy()

Insights

    The function iterates over reflected fields and checks each one for Copyable conformance, skipping any field that cannot be copied.
    As a method, copy_to() does not require a type parameter such as copy_to[T](). It has direct access to Self, which is enabled by MakeCopyable trait adoption.
    The copy_to() implementation is heavily parameterized and evaluated at compile time. It uses comptime for and comptime if together with reflection calls.
    The __struct_field_ref(idx, self) and __struct_field_ref(idx, other) calls return references to fields by index, which allows both reading from the source instance and writing to the destination instance.

Create a struct

The following MultiType struct conforms to MakeCopyable meaning you can call copy_to on its instances:

@fieldwise_init
struct MultiType(Writable, MakeCopyable):
    var w: String
    var x: Int
    var y: Bool
    var z: Float64

    def write_to[W: Writer](self, mut writer: W):
        writer.write("[{}, {}, {}, {}]".format(self.w, self.x, self.y, self.z))

Use the copying functionality

Demonstrating the behavior, this main() function creates two instances, one populated by normal values, and one essentially zeroed-out. After copying, target_instance has received its values from original_instance's fields:

def main():
    var original_instance = MultiType("Hello", 1, True, 2.5)
    var target_instance = MultiType("", 0, False, 0.0)

    print("Original instance:", original_instance)     # "Hello", 1, True, 2.5
    print("Target instance before: ", target_instance) # "", 0, False, 0.0

    original_instance.copy_to(target_instance)
    print("Target instance after: ", target_instance)  #  "Hello", 1, True, 2.5

Example: Testing equality

This example demonstrates how reflection can be used to implement structural equality. Compile-time loop unrolling, conformance checks, and reflection drive runtime comparisons. This is the same pattern used for generic equality, hashing, and validation logic.

In each iteration, test_equality checks for Equatable conformance and retrieves field value references from the lhs and rhs arguments. It uses early return for the first inequality (False), otherwise returning True:

from std.reflection import struct_field_count, struct_field_types

def test_equality[T: AnyType](lhs: T, rhs: T) -> Bool:
    comptime field_count = struct_field_count[T]()
    comptime field_types = struct_field_types[T]()

    comptime for idx in range(field_count):
        # Guard: field type must be equatable
        comptime field_type = field_types[idx]
        comptime if not conforms_to(field_type, Equatable): continue

        # Fetch values
        ref lhs_value = __struct_field_ref(idx, lhs)
        ref rhs_value = __struct_field_ref(idx, rhs)

        # Early exit with `False` when inequality found
        if trait_downcast[Equatable](lhs_value) != trait_downcast[Equatable](rhs_value): return False

    return True

Insights

    This example uses two key elements to ensure smooth operation: conforms_to() and trait_downcast[]().
    conforms_to() ensures each field's type is Equatable and therefore works with the inequality operator (!=).
    trait_downcast is used with parameter input types. It rebinds a typed value, returning a value that conforms to the specified Trait. If its type, after resolution, does not conform to that trait, it will produce a compilation error. Checking for conformance first ensures that error won't occur.
    As its name suggests, __struct_field_ref() is limited to use with struct types.

Calling the tests

To demonstrate this function, the following main() first copies values (equal), and then mutates original_instance and tests again (unequal):

def main():
    var original_instance = MultiType("Hello", 1, True, 2.5)
    var target_instance = MultiType("", 0, False, 0.0)
    original_instance.copy_to(target_instance)
    print("Values equal" if \
        test_equality(original_instance, target_instance) \
        else "Values not equal") # Values equal


    original_instance.z = 42.0
    print("Values equal" if \
        test_equality(original_instance, target_instance) \
        else "Values not equal") # Values not equal

Advanced default method patterns

The "Default method implementations" section above shows simple defaults like printing a string or negating an equality check. In practice, trait defaults are far more powerful — a default method body is ordinary Mojo code and can do anything a regular method can. This section documents patterns that go beyond simple one-liners.
Default methods calling abstract methods

A default method can call other methods required by the same trait, even if those methods are abstract (not yet implemented). The compiler guarantees that any conforming struct will have provided an implementation, so the call is always valid at monomorphization time.

```mojo
trait Codec:
    def raw_encode(self, data: Span[Byte]) -> List[Byte]: ...
    def raw_decode(self, data: Span[Byte]) -> List[Byte]: ...

    def encode_string(self, text: String) -> String:
        var encoded = self.raw_encode(text.as_bytes())
        return String(unsafe_from_utf8=Span(encoded))

    def decode_string(self, text: String) -> String:
        var decoded = self.raw_decode(text.as_bytes())
        return String(unsafe_from_utf8=Span(decoded))
```

Here `encode_string` and `decode_string` are defaults that call the abstract `raw_encode` and `raw_decode`. A conformer only needs to implement the two raw methods and gets the string-level API for free:

```mojo
struct Base64Codec(Codec):
    def raw_encode(self, data: Span[Byte]) -> List[Byte]:
        # ... base64 encoding logic
    def raw_decode(self, data: Span[Byte]) -> List[Byte]:
        # ... base64 decoding logic
    # encode_string and decode_string inherited
```

Default methods with comptime members

Defaults can reference comptime members declared in the same trait using `Self.MEMBER`. Combined with calling abstract methods, this lets you write shared logic that varies based on both comptime data and runtime behavior provided by the conformer:

```mojo
trait ArchPrimitives:
    comptime NR_write: Int
    comptime NR_exit: Int
    def syscall[count: Int](self, nr: Int, *args: Int) -> Int: ...

trait Syscalls(ArchPrimitives):
    def sys_write(self, fd: Int, buf: Int, count: Int) -> Int:
        return self.syscall[3](Self.NR_write, fd, buf, count)

    def sys_exit(self, code: Int = 0):
        _ = self.syscall[1](Self.NR_exit, code)
```

Each conformer provides its own `NR_write`, `NR_exit`, and `syscall` implementation. The shared wrappers `sys_write` and `sys_exit` compose these primitives into a portable interface without any manual forwarding.

The subtrait template method pattern

The previous example demonstrates a powerful structural pattern: split a trait hierarchy into an abstract "primitives" trait and a "shared behavior" subtrait whose defaults build on those primitives.

```
ArchPrimitives (abstract: what varies per implementation)
    └── Syscalls(ArchPrimitives) (defaults: shared logic built on the primitives)
```

This is the trait-level equivalent of the template method pattern. The parent trait defines the varying steps. The subtrait provides the algorithm as defaults. Conformers implement the parent and inherit the subtrait's shared behavior automatically.

This pattern is most useful when:

- Multiple implementations share significant common logic (the defaults).
- The implementations differ in a small number of core operations (the abstract methods).
- The shared logic calls the varying operations — it's not just data access, but behavioral composition.

For example, an x86_64 implementation and an aarch64 implementation might have different syscall register ABIs and different signal handling, but share identical logic for memory mapping, futex operations, and file I/O:

```mojo
struct X86_64(Syscalls):
    comptime NR_write = 1
    comptime NR_exit = 60
    def syscall[count: Int](self, nr: Int, *args: Int) -> Int:
        # x86_64 register ABI via inline assembly
        ...

struct AArch64(Syscalls):
    comptime NR_write = 64
    comptime NR_exit = 93
    def syscall[count: Int](self, nr: Int, *args: Int) -> Int:
        # aarch64 register ABI via inline assembly
        ...
```

Both implementations provide ~3 things (syscall numbers + the raw mechanism) and inherit ~20 shared wrappers from the `Syscalls` defaults. Without this pattern, each implementation would need to manually duplicate all the shared wrappers, or a forwarding wrapper would need to delegate every method individually.

Default methods calling free functions

Default method bodies can call any function in scope, not just methods on `self` or `Self.MEMBER` accesses. This includes free functions from other modules:

```mojo
# In transform_utils.mojo
def bytes_to_text(data: Span[Byte]) -> String:
    # ... conversion logic

def text_to_bytes(text: String) -> List[Byte]:
    # ... conversion logic

# In capabilities.mojo
from .transform_utils import bytes_to_text, text_to_bytes

trait ByteTransform(Movable, ImplicitlyDestructible):
    def encode(self, data: Span[Byte]) -> String:
        return bytes_to_text(data)

    def decode(self, text: String) -> List[Byte]:
        return text_to_bytes(text)
```

Any struct conforming to `ByteTransform` inherits both methods without writing any code. A conformer can still override a default if it needs different behavior:

```mojo
struct StandardTransform(ByteTransform):
    def __init__(out self): pass
    # encode and decode inherited from defaults

struct CustomTransform(ByteTransform):
    def __init__(out self): pass

    def encode(self, data: Span[Byte]) -> String:
        # Custom encoding that overrides the default
        ...
    # decode still inherited from default
```

Complex default method bodies

Default methods are not limited to one-liners. They can contain local variables, control flow, pointer operations, and any other valid Mojo code:

```mojo
trait QueryableMemory:
    def syscall[count: Int](self, nr: Int, *args: Int) -> Int: ...
    comptime NR_move_pages: Int

    def query_page_node(self, addr: Int) -> Int:
        var pages: InlineArray[Int, 1] = [addr]
        var status: InlineArray[Int32, 1] = [Int32(-1)]
        var pages_ptr = UnsafePointer(to=pages)
        var status_ptr = UnsafePointer(to=status)
        var result = self.syscall[6](
            Self.NR_move_pages, 0, 1,
            Int(pages_ptr), 0, Int(status_ptr), 0,
        )
        _ = pages_ptr[]
        _ = status_ptr[]
        if result < 0:
            return result
        return Int(status[0])
```

This default method declares local `InlineArray` values, creates `UnsafePointer` references, calls an abstract method with compile-time parameters, and includes conditional control flow. Any struct that conforms to `QueryableMemory` inherits this complete implementation.

When to use trait defaults vs. other patterns

Trait defaults are the right choice when:

- Multiple conformers would have identical implementations for a method.
- The shared logic depends on abstract methods or comptime members that vary per conformer.
- You want to eliminate a forwarding wrapper that manually delegates every method.

Trait defaults are not useful when:

- There's only one conformer — a direct implementation is simpler.
- The method logic doesn't depend on anything that varies — make it a free function instead.
- The "shared" behavior actually differs subtly between conformers — forced sharing creates bugs.

Linear types with @explicit_destroy

By default, structs conform to `ImplicitlyDestructible`, meaning the compiler automatically destroys them when their lifetime ends. Traits do not inherit `ImplicitlyDestructible` by default.

Applying `@explicit_destroy` to a struct opts it out of `ImplicitlyDestructible`. The compiler no longer inserts automatic destruction for values of that type. Instead, the only way to consume an `@explicit_destroy` value is through a `deinit self` method. If a value of an explicitly destroyed type is not consumed, the compiler emits an error.

This makes `@explicit_destroy` types linear types: every value must be used exactly once. This is useful for modeling resources that require explicit cleanup, synchronization tokens, or ownership transfer protocols where silent destruction would be a bug.

```mojo
@explicit_destroy
struct PoolFence:
    """A synchronization token that must be explicitly returned to the pool."""
    var _pool_id: Int
    var _fence_id: Int

    def __init__(out self, pool_id: Int, fence_id: Int):
        self._pool_id = pool_id
        self._fence_id = fence_id

    def wait_and_release(deinit self):
        """Block until the fence signals, then release it back to the pool."""
        _sync_fence(self._pool_id, self._fence_id)

    def release(deinit self):
        """Release the fence without waiting."""
        _release_fence(self._pool_id, self._fence_id)
```

Using the `PoolFence`:

```mojo
def submit_work(pool: Pool) raises:
    var fence = pool.submit(work_item)  # Returns a PoolFence
    # ... do other work ...
    fence.wait_and_release()  # Consumes the fence — required

    # Forgetting to consume the fence is a compile error:
    # var fence2 = pool.submit(work_item2)
    # return  # ERROR: unconsumed value of type 'PoolFence'
```

Key points:

- `@explicit_destroy` on a struct removes its implicit `ImplicitlyDestructible` conformance.
- The struct must provide at least one `deinit self` method to consume values.
- Unconsumed values produce a compile-time error.
- This pattern is well-suited for synchronization tokens, file handles that must be flushed, transaction objects that must be committed or rolled back, and any resource where silent drop would be incorrect.

Learn more

    Visit the reflection package documentation to explore additional Mojo reflection capabilities.
    Learn more about traits.
    (Coming soon): Dive into generics, trait conformance tests with conforms_to(), and downcasts with trait_downcast[]().
