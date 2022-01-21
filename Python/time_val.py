from math import exp

def future_discrete_value(x, r, n):
    return x*((1+r)**n)

def present_discrete_value(x, r, n):
    return x/((1+r)**n)

def future_continuous_value(x, r, t):
    return x*exp(r*t)

def present_continuous_value(x, r, t):
    return x*exp(-r*t)

if __name__ == '__main__':
    #val of investment
    x=100
    r=0.05
    n=5

    print("Future discrete value:" , future_discrete_value(x,r,n))
    print("Future continuous value:" , future_continuous_value(x,r,n))
    print("\nPresent discrete value:" , present_discrete_value(x,r,n))
    print("Present continuous value:" , present_continuous_value(x,r,n))