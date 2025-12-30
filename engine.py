import math

class Value:
    def __init__(self,data,_children=(),_op='',label=''):
        self._op = _op
        self.data=data
        self._backward=lambda: None
        self._prev =set(_children)
        self.label=label
        self.grad =0.0
    def __repr__(self):
        return f'Value(data= {self.data})'

    def __add__(self,other):
        out = Value(self.data + other.data,(self,other),'+')
        def _backward():
            self.grad=1.0*out.grad
            other.grad= 1.0*out.grad

        out._backward=_backward
        return out

    def __mul__(self,other):
        out = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad=(other.data*out.grad)
            other.grad=(self.data)*out.grad

        out._backward= _backward
        return out
    def tanh(self):
        n=self.data
        t=(math.exp(2*n)-1)/(math.exp(2*n)+1)
        out=Value(t,(self,),'tanh')
        def _backward():
            self.grad=out.grad*(1-t**2)


        out._backward= _backward
        return out





