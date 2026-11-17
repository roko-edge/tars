use std::error::Error;
use tars::{view::graph::NetGraph, *};

fn main() -> Result<(), Box<dyn Error>> {
    let mut train = experiments::or::train();

    println!("Starting {:?} experiment...", train.experiment_type);
    println!("Check docs/experiments/ to view experiments API.");

    let step = (train.epochs / 20).max(1);
    let optimizer = BGD::new(train.lr);
    let mut prev_cost = cost(&train.model, &train.dataset);
    println!("epoch: 000000, cost is:{:014.8}", prev_cost);

    for i in 1..=train.epochs {
        let grad = backward(&train.model, &train.dataset);
        optimizer.step(&mut train.model, &grad);
        let curr_cost = cost(&train.model, &train.dataset);
        if i % step == 0 {
            println!(
                "epoch: {:06.0}, cost is:{:014.8}, {:07.3}% better",
                i,
                curr_cost,
                ((prev_cost - curr_cost) * 100.0) / prev_cost
            );
            prev_cost = curr_cost;
        }
    }

    for d in &train.dataset {
        println!(
            "Recived input: {:?}; The network returns: {:?};  Expected: {:?} ",
            d.input,
            train.model.forward(&d.input),
            d.target
        );
    }

    // export_model(&train.model)?;
    // export_data(&train.dataset)?;

    // Export the network topology to a .png
    let mut net = NetGraph::new(train.model, &train.dataset[0].input);
    net.network_to_graph()?;
    net.plot();
    Ok(())
}
