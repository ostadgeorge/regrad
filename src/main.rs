use regrad::Value;


fn main() {
    println!("Hello, world!");

    let v1 = Value::from(1.2);
    let v2 = Value::from(3.4);

    // let v3 = &v1 + &v2;
    // dbg!(v3.data());
    // v3.backward();
    // dbg!(v1.gradient());
    // dbg!(v2.gradient());
    // dbg!(v3.gradient());

    let v4 = &(&v1 * &v1) * &v2;

    dbg!(v4.data());
    assert_eq!(v4.data(), 4.896);

    v4.backward();
    dbg!(v1.gradient());
    dbg!(v2.gradient());
    dbg!(v4.gradient());

    assert_eq!(v1.gradient(), 8.16);
    assert_eq!(v2.gradient(), 1.44);
    assert_eq!(v4.gradient(), 1.0);

}
