use std::error::Error;
use std::time::Instant;
use tars::tensor_core::MatMul;
use tars::*;

fn main() -> Result<(), Box<dyn Error>> {
    let t1 = Tensor::<f32>::from_vec((0..1000000).map(|i| i as f32).collect(), vec![1000, 1000]);

    let t2 = Tensor::<f32>::from_vec((0..1000000).map(|i| i as f32).collect(), vec![1000, 1000]);

    let fast_matmul_begin = Instant::now();
    t1.matmul(&t2);
    let fast_matmul_end = fast_matmul_begin.elapsed().as_millis();
    println!("fast matmul elapsed time in ms:{:?}", fast_matmul_end);

    let std_matmul_begin = Instant::now();
    t1.std_matmul(&t2);
    let std_matmul_end = std_matmul_begin.elapsed().as_millis();
    println!("standard matmul elapsed time in ms:{:?}", std_matmul_end);

    Ok(())
}
