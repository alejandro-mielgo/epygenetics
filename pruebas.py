def f(x,y)->float:
    return x + y

def change_sign(func):
    def wrapper(x,y):
        return -1*func(x,y)
    return wrapper


print(f(2,1))

@change_sign
def fun(x,y)->float:
    return x + y

print(fun(2,1))