use std::{cell::RefCell, rc::Rc};

use crate::Value;

pub struct Tensor {
    internal: Rc<RefCell<TensorInternal>>,
}

impl Tensor {
    pub fn new(data: Vec<Value>, shape: Vec<usize>) -> Tensor {
        let size = shape.iter().product();
        let strides = shape
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
            .collect();

        Tensor {
            internal: Rc::new(RefCell::new(TensorInternal::new(
                data, shape, strides, size,
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

    pub fn reshape(&self, shape: Vec<usize>) -> Tensor {
        assert_eq!(self.size(), shape.iter().product());
        Tensor::new(self.data(), shape)
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
