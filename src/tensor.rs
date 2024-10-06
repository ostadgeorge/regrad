// implemented from:
// https://towardsdatascience.com/recreating-pytorch-from-scratch-with-gpu-support-and-automatic-differentiation-8f565122a3cc

use std::{cell::RefCell, hash::Hash, ops::{Add, Deref, Mul, Neg, Sub}, rc::Rc};

use crate::Value;

pub struct Tensor {
    internal: Rc<RefCell<TensorInternal>>,
}

impl Tensor {
    pub fn new(data: Vec<Value>, shape: Vec<usize>) -> Tensor {
        let size = shape.iter().product();
        let strides = compute_strides(shape.clone());

        Tensor {
            internal: Rc::new(RefCell::new(TensorInternal::new(
                data, shape, strides, size,
            ))),
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Tensor {
        let size = shape.iter().product();
        let strides = compute_strides(shape.clone());

        Tensor {
            internal: Rc::new(RefCell::new(TensorInternal::new(
                vec![Value::from(0.0); size],
                shape,
                strides,
                size,
            ))),
        }
    }

    pub fn ones(shape: Vec<usize>) -> Tensor {
        let size = shape.iter().product();
        let strides = compute_strides(shape.clone());

        Tensor {
            internal: Rc::new(RefCell::new(TensorInternal::new(
                vec![Value::from(1.0); size],
                shape,
                strides,
                size,
            ))),
        }
    }

    pub fn data(&self) -> Vec<Value> {
        self.internal.borrow().data.clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.internal.borrow().shape.clone()
    }

    pub fn strides(&self) -> Vec<usize> {
        self.internal.borrow().strides.clone()
    }

    pub fn size(&self) -> usize {
        self.internal.borrow().size
    }

    pub fn reshape(&self, shape: Vec<usize>) -> &Tensor {
        assert_eq!(self.size(), shape.iter().product());
        self.internal.borrow_mut().shape = shape;

        self
    }

    pub fn gradient(&self) -> Tensor {
        let data = self
            .data()
            .iter()
            .map(|v| Value::from(v.gradient()))
            .collect();
        let shape = self.shape();
        let strides = self.strides();
        let size = self.size();

        Tensor {
            internal: Rc::new(RefCell::new(TensorInternal::new(
                data, shape, strides, size,
            ))),
        }
    }

    pub fn zero_grad(&self) {
        for v in self.data() {
            v.zero_grad();
        }
    }

    pub fn update(&self, factor: f64) {
        for v in self.data() {
            v.update(factor);
        }
    }

    pub fn backward(&self) {
        unimplemented!("Tensor backward")
    }
}

impl Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.internal.borrow().hash(state);
    }
}

impl Deref for Tensor {
    type Target = Rc<RefCell<TensorInternal>>;

    fn deref(&self) -> &Self::Target {
        &self.internal
    }
}

fn add(u: &Tensor, v: &Tensor) -> Tensor {
    assert_eq!(u.shape(), v.shape());

    let data = u
        .data()
        .iter()
        .zip(v.data().iter())
        .map(|(u, v)| u + v)
        .collect();

    let shape = u.shape();
    let size = u.size();
    let strides = u.strides();

    Tensor {
        internal: Rc::new(RefCell::new(TensorInternal::new(
            data, shape, strides, size,
        ))),
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, other: Tensor) -> Tensor {
        add(&self, &other)
    }
}

impl<'a, 'b> Add<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn add(self, other: &'b Tensor) -> Tensor {
        add(self, other)
    }
}

fn mul(u: &Tensor, v: &Tensor) -> Tensor {
    assert_eq!(u.shape(), v.shape());

    let data = u
        .data()
        .iter()
        .zip(v.data().iter())
        .map(|(u, v)| u * v)
        .collect();

    let shape = u.shape();
    let size = u.size();
    let strides = u.strides();

    Tensor {
        internal: Rc::new(RefCell::new(TensorInternal::new(
            data, shape, strides, size,
        ))),
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, other: Tensor) -> Tensor {
        mul(&self, &other)
    }
}

impl<'a, 'b> Mul<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn mul(self, other: &'b Tensor) -> Tensor {
        mul(self, other)
    }
}

impl Mul<Value> for Tensor {
    type Output = Tensor;

    fn mul(self, other: Value) -> Tensor {
        let tmp_tensor = Tensor::new(vec![other; self.size()], self.shape());
        mul(&self, &tmp_tensor)
    }
}

impl<'a> Mul<&'a Value> for &Tensor {
    type Output = Tensor;

    fn mul(self, other: &'a Value) -> Tensor {
        let tmp_tensor = Tensor::new(vec![other.clone(); self.size()], self.shape());
        mul(self, &tmp_tensor)
    }
}

impl Neg for Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self * Value::from(-1.0)
    }
}

impl<'a> Neg for &'a Tensor {
    type Output = Tensor;

    fn neg(self) -> Tensor {
        self * &Value::from(-1.0)
    }
}

impl Sub for Tensor {
    type Output = Tensor;

    fn sub(self, other: Tensor) -> Tensor {
        self + (-other)
    }
}

impl<'a, 'b> Sub<&'b Tensor> for &'a Tensor {
    type Output = Tensor;

    fn sub(self, other: &'b Tensor) -> Tensor {
        self + &(-other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let t1 = Tensor::new(vec![Value::from(1.0), Value::from(2.0)], vec![2]);
        let t2 = Tensor::new(vec![Value::from(3.0), Value::from(4.0)], vec![2]);

        let t3 = &t1 + &t2;

        assert_eq!(t3.data().iter().map(|v| v.data()).collect::<Vec<f64>>(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_mul() {
        let t1 = Tensor::new(vec![Value::from(1.0), Value::from(2.0)], vec![2]);
        let t2 = Tensor::new(vec![Value::from(3.0), Value::from(4.0)], vec![2]);

        let t3 = &t1 * &t2;

        assert_eq!(t3.data().iter().map(|v| v.data()).collect::<Vec<f64>>(), vec![3.0, 8.0]);
    }

    #[test]
    fn test_mul_value() {
        let t1 = Tensor::new(vec![Value::from(1.0), Value::from(2.0)], vec![2]);
        let v1 = Value::from(3.0);

        let t2 = &t1 * &v1;

        assert_eq!(t2.data().iter().map(|v| v.data()).collect::<Vec<f64>>(), vec![3.0, 6.0]);
    }

    #[test]
    fn test_neg() {
        let t1 = Tensor::new(vec![Value::from(1.0), Value::from(2.0)], vec![2]);

        let t2 = -t1;

        assert_eq!(t2.data().iter().map(|v| v.data()).collect::<Vec<f64>>(), vec![-1.0, -2.0]);
    }

    #[test]
    fn test_sub() {
        let t1 = Tensor::new(vec![Value::from(1.0), Value::from(2.0)], vec![2]);
        let t2 = Tensor::new(vec![Value::from(3.0), Value::from(4.0)], vec![2]);

        let t3 = &t1 - &t2;

        assert_eq!(t3.data().iter().map(|v| v.data()).collect::<Vec<f64>>(), vec![-2.0, -2.0]);
    }
}

#[derive(Clone, Eq, PartialEq, Debug, Hash)]
pub struct TensorInternal {
    data: Vec<Value>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    size: usize,
}

impl TensorInternal {
    pub fn new(
        data: Vec<Value>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        size: usize,
    ) -> TensorInternal {
        TensorInternal {
            data,
            shape,
            strides,
            size,
        }
    }
}

fn compute_strides(shape: Vec<usize>) -> Vec<usize> {
    shape
        .iter()
        .rev()
        .skip(1)
        .fold(vec![1], |mut acc, &s| {
            acc.push(acc.last().unwrap() * s);
            acc
        })
        .iter()
        .rev()
        .cloned()
        .collect()
}
