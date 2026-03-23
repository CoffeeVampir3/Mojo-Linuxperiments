def call_it[f: def(Int) -> Int](x: Int) -> Int:
    return f(x)

def double(x: Int) -> Int:
    return x * 2

def main():
    print(call_it[double](5))
