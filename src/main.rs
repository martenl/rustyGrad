use std::borrow::BorrowMut;
use std::fmt::{Display, Formatter};
use std::ops::{Add, Sub, Mul, Div, Neg};

fn main() {
    let a = Value::from(-4.0);
    let b = Value::from(2.0);
    let c = &a + &b;
    let d = &a * &b + b.pow(3.0);
    let e = c.relu();
    println!("{}", a);
    println!("{}", d);
    println!("{}", e);

    let lambda = |x:f32| {
        x + 1.0
    };

    let z = Value::from(-4.0);
    println!("a == z? {}", a==z);
}

#[derive(Debug, Clone, PartialEq)]
enum Operation {
    Scalar, Add, Neg, Mul, Div, Pow, Relu
}

fn op_to_lambda(op: Operation) -> fn(f32, f32) -> f32 {
    match op {
        Operation::Scalar => |x:f32, y:f32| x + y,
        Operation::Add => |x:f32, y:f32| x + y,
        Operation::Mul => |x:f32, y:f32| x * y,
        Operation::Neg => |x:f32, y:f32| -x,
        Operation::Div => |x:f32, y:f32| x / y,
        Operation::Pow => |x:f32, y:f32| x.powf(y),
        Operation::Relu => |x:f32, y:f32| if(x < 0.0) {x} else {-x} ,
    }
}

impl Display for Operation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[derive(Clone, PartialEq)]
struct Value {
    data: f32,
    grad: f32,
    op: Operation,
    children: Vec<Value>
}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value(data:{},operation:{})", self.data, self.op.to_string())
    }
}


impl Value {

    //fn new(_data:f32, _chi)
    fn from(data:f32) -> Value {
        Value{ data, grad:0.0, op: Operation::Scalar, children:vec![]}
    }

    fn pow(self, exp:f32) -> Value {
        Value{ data:self.data.powf(exp), grad: 0.0, op: Operation::Pow, children: vec![self, Value::from(exp)]}
    }

    fn relu(self) -> Value {
        Value {
            data: if self.data < 0.0 { 0.0 } else { self.data},
            grad: 0.0,
            op: Operation::Relu,
            children: vec![self]
        }
    }

    fn backward(mut self) {
        self.grad = 1.0;
    }

    fn backward_step(mut self) {
        match self.op {
            Operation::Scalar => {}
            Operation::Add => {
                let mut children = self.children;
                for child in children.iter_mut() {
                    child.grad += self.grad;
                }
            }
            Operation::Neg => {}
            Operation::Mul => {
                let mut children = self.children;
                children[0].grad += self.grad * children[1].data;
                children[1].grad += self.grad * children[0].data;
                /*for (idx, child) in children.iter_mut().enumerate() {
                    for (other_idx, other_child) in self.children.iter().enumerate() {
                        if idx != other_idx {
                            child.grad += (self.grad * other_child.data);
                        }
                    }
                }}*/
            },
            Operation::Div => {}
            Operation::Pow => {
                let mut children = self.children.iter_mut();
                let mut child = children.next().unwrap();
                let mut exp = children.next().unwrap();
                child.data += (exp.data * child.data.powf(exp.data-1.0)) * self.grad;
            }
            Operation::Relu => {
                let mut children = self.children;
                for child in children.iter_mut() {
                    child.grad += if self.data > 0.0 { self.grad } else { 0.0 };
                }
            }
        }
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        Value {
            data: self.data + rhs.data,
            grad: 0.0,
            op: Operation::Add,
            children: vec![self, rhs]
        }
    }
}

impl Add for &Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        Value {
            data: self.data + rhs.data,
            grad: 0.0,
            op: Operation::Add,
            children: vec![(*self).clone(), (*rhs).clone()]
        }
    }
}
/*
impl Add<&Value> for f32 {
    type Output = Value;

    fn add(self, rhs: &Value) -> Self::Output {
        Value {
            data: self + rhs.data,
            grad: 0.0,
            children: vec![Value::from(self), *rhs]
        }
    }
}
*/
impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}

/*impl Sub<&Value> for f32{
    type Output = Value;

    fn sub(self, rhs: &Value) -> Self::Output {
        self + -rhs
    }
}*/

impl Mul for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        Value {
            data: self.data * rhs.data,
            grad: 0.0,
            op: Operation::Mul,
            children: vec![self, rhs]
        }
    }
}

impl Mul for &Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        Value {
            data: self.data * rhs.data,
            grad: 0.0,
            op: Operation::Mul,
            children: vec![(*self).clone(), (*rhs).clone()]
        }
    }
}
/*
impl Mul<&Value> for f32 {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Self::Output {
        Value {
            data: self * rhs.data,
            grad: 0.0,
            children: vec![Value::from(self), *rhs]
        }
    }
}
*/
impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        Value{data: self.data*-1.0 , grad: 0.0, op: Operation::Neg, children: vec![self]}
    }
}

impl Neg for &Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        Value{data: self.data*-1.0 , grad: 0.0, op: Operation::Neg, children: vec![(*self).clone()]}
    }
}