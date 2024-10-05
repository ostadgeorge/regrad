# regrad
backtrack in rust

## Usage
```rust
    let v1 = Value::from(1.2);
    let v2 = Value::from(3.4);
    let v3 = &(&v1 * &v1) * &v2;

    dbg!(v3.data());
    assert_eq!(v3.data(), 4.896);

    v3.backward();
    dbg!(v1.gradient());
    dbg!(v2.gradient());
    dbg!(v3.gradient());

    assert_eq!(v1.gradient(), 8.16);
    assert_eq!(v2.gradient(), 1.44);
    assert_eq!(v3.gradient(), 1.0);
```