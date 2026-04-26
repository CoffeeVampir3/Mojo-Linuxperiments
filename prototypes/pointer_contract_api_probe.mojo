from std.memory import Pointer, UnsafePointer


@fieldwise_init
struct Token(Movable):
    var value: Int

    @always_inline
    def get[origin: MutOrigin](ref [origin] self) -> Int:
        return self.value


@fieldwise_init
struct TokenContract[origin: MutOrigin](Copyable, ImplicitlyCopyable):
    var token: Pointer[Token, Self.origin]

    @always_inline
    def read(self) -> Int:
        return self.token[].get()


def main():
    var token = Token(7)
    var contract = TokenContract(Pointer(to=token))
    print(contract.read())
