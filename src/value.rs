use std::{
    cell::{Ref, RefCell},
    fmt::{Debug, Formatter, Result},
    hash::Hash,
    ops::{Add, Deref, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Value {
    internal: Rc<RefCell<ValueInternal>>,
}

impl Value {
    pub fn from<T>(t: T) -> Value
    where
        T: Into<Value>,
    {
        t.into()
    }

    fn new(internal: ValueInternal) -> Value {
        Value {
            internal: Rc::new(RefCell::new(internal)),
        }
    }

    pub fn data(&self) -> f64 {
        self.internal.borrow().data
    }

    pub fn gradient(&self) -> f64 {
        self.internal.borrow().gradient
    }

    pub fn zero_grad(&self) {
        self.internal.borrow_mut().gradient = 0.0;
    }

    pub fn update(&self, factor: f64) {
        let gradient = self.internal.borrow().gradient;
        self.internal.borrow_mut().data += factor * gradient;
    }

    pub fn backward(&self) {
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(self.clone());

        self.internal.borrow_mut().gradient = 1.0;

        while let Some(value) = queue.pop_front() {
            let internal = value.internal.borrow();
            if visited.contains(&value) {
                continue;
            }
            visited.insert(value.clone());

            if let Some(propagate) = internal.propagate {
                propagate(&internal);
            }

            for previous in internal.previous.iter() {
                queue.push_back(previous.clone());
            }
        }
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.internal.borrow().hash(state);
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<ValueInternal>>;

    fn deref(&self) -> &Self::Target {
        &self.internal
    }
}

impl From<f64> for Value {
    fn from(data: f64) -> Self {
        Value::new(ValueInternal::new(data, None, None, vec![], None))
    }
}

fn add(u: &Value, v: &Value) -> Value {
    let data = u.data() + v.data();
    let propagate: BackPropagteFn = |value: &Ref<ValueInternal>| {
        let gradient = value.gradient;

        value.previous[0].internal.borrow_mut().gradient += gradient;
        value.previous[1].internal.borrow_mut().gradient += gradient;
    };

    Value::new(ValueInternal::new(
        data,
        None,
        Some(Operation::Add),
        vec![u.clone(), v.clone()],
        Some(propagate),
    ))
}

impl Add for Value {
    type Output = Value;

    fn add(self, rhs: Self) -> Self::Output {
        add(&self, &rhs)
    }
}

impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;

    fn add(self, rhs: &'b Value) -> Self::Output {
        add(self, rhs)
    }
}

fn mul(u: &Value, v: &Value) -> Value {
    let data = u.data() * v.data();

    let propagate: BackPropagteFn = |value: &Ref<ValueInternal>| {
        let ud = value.previous[0].internal.borrow().data;
        let vd = value.previous[1].internal.borrow().data;

        value.previous[0].internal.borrow_mut().gradient += value.gradient * vd;
        value.previous[1].internal.borrow_mut().gradient += value.gradient * ud;
    };

    Value::new(ValueInternal::new(
        data,
        None,
        Some(Operation::Mul),
        vec![u.clone(), v.clone()],
        Some(propagate),
    ))
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, rhs: Self) -> Self::Output {
        mul(&self, &rhs)
    }
}

impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;

    fn mul(self, rhs: &'b Value) -> Self::Output {
        mul(self, rhs)
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        mul(&self, &Value::from(-1.0))
    }
}

impl<'a> Neg for &'a Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        mul(self, &Value::from(-1.0))
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, rhs: Self) -> Self::Output {
        add(&self, &(-rhs))
    }
}

impl<'a, 'b> Sub<&'b Value> for &'a Value {
    type Output = Value;

    fn sub(self, rhs: &'b Value) -> Self::Output {
        add(self, &(-rhs))
    }
}

type BackPropagteFn = fn(value: &Ref<ValueInternal>);

#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub enum Operation {
    Add,
    Sub,
    Mul,
}

#[derive(Clone)]
pub struct ValueInternal {
    data: f64,
    gradient: f64,
    label: Option<String>,
    operation: Option<Operation>,
    previous: Vec<Value>,
    propagate: Option<BackPropagteFn>,
}

impl ValueInternal {
    pub fn new(
        data: f64,
        label: Option<String>,
        operation: Option<Operation>,
        previous: Vec<Value>,
        propagate: Option<BackPropagteFn>,
    ) -> ValueInternal {
        ValueInternal {
            data,
            gradient: 0.0,
            label,
            operation,
            previous,
            propagate,
        }
    }
}

impl PartialEq for ValueInternal {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
            && self.gradient == other.gradient
            && self.label == other.label
            && self.operation == other.operation
            && self.previous == other.previous
    }
}

impl Eq for ValueInternal {}

impl Hash for ValueInternal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.to_bits().hash(state);
        self.gradient.to_bits().hash(state);
        self.label.hash(state);
        self.operation.hash(state);
        self.previous.hash(state);
    }
}

impl Debug for ValueInternal {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(
            f,
            "ValueInternal {{ data: {}, gradient: {}, label: {:?}, operation: {:?}, previous: {:?} }}",
            self.data, self.gradient, self.label, self.operation, self.previous
        )
    }
}