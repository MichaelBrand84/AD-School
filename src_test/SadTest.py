from floatsad import FloatSad
import mathsad



def f(arg:float) -> FloatSad:
    x = FloatSad(arg)
    b1 = x ** 2
    b2 = mathsad.atanh(b1)
    return b1



def main():
    y = f(0.2)
    print(y.derivative == 2*0.2)




if __name__ == '__main__': 
    print("\n\n")
    main()
    print("\n\n")
