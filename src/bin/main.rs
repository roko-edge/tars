use std::error::Error;
use tars::*;

fn main() -> Result<(), Box<dyn Error>> {
    let mut train = experiments::xor::train();

    println!("Starting {:?} experiment...", train.experiment_type);
    println!("Check docs/experiments/ to view experiments API.");

    let step = (train.epochs / 20).max(1);
    let optimizer = BGD::new(train.lr);
    let mut prev_cost = cost(&train.model, &train.dataset);
    println!("epoch: 000000, cost is:{:014.8}", prev_cost);

    let mut loader = DataMiniBatches::new(&train.dataset, 1, 42);

    for i in 1..=train.epochs {
        for batch in loader.iter_epoch() {
            let cache = train.model.forward_cache(&batch.input);
            let grad = backward(&train.model, &cache, &batch.target);
            optimizer.step(&mut train.model, &grad);
        }

        if i % step == 0 {
            let curr_cost = cost(&train.model, &train.dataset);

            println!(
                "epoch: {:06.0}, cost is:{:014.8}, {:07.3}% better",
                i,
                curr_cost,
                ((prev_cost - curr_cost) * 100.0) / prev_cost
            );
            prev_cost = curr_cost;
        }
    }

    let scales;
    (train.model, scales) = train.model.quantize();
    export_model(&train.model, &scales)?;
    export_inference(&train.model, &train.dataset)?;

    // Export the network topology to a .png
    // let mut net = NetGraph::new(train.model, &train.dataset[0].input);
    // net.network_to_graph()?;
    // net.plot();
    Ok(())
}
