use tars::view::graph::*;
use tars::{Tensor, sequential::*};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Sequential::new(1).linear(2).relu().linear(3).relu();
    let input = vec![67.67];

    let net = NetGraph::new(model, &input);
    println!("Network topology:");
    println!("{:?}", net);

    Ok(())
}
