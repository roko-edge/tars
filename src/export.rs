use crate::{Data, modules::*};
use std::error::Error;
use std::fs::File;
use std::io::Write;

pub fn to_q8_24(x: f32) -> u32 {
    //WARNING: Rust conversion from f32 to u32 saturate in type limits
    ((x * 16_777_216.0) as i32) as u32
}

pub fn export_model(model: &Sequential) -> Result<(), Box<dyn Error>> {
    let mut file_weights = File::create("npu/data/weights.mem")?;
    let mut file_bias = File::create("npu/data/bias.mem")?;
    for module in &model.modules {
        match module {
            AnyModule::Linear(linear) => {
                for o in 0..linear.weights.len() {
                    for weight in &linear.weights[o] {
                        writeln!(file_weights, "{:08X}", to_q8_24(*weight))?;
                    }
                    writeln!(file_bias, "{:08X}", to_q8_24(linear.bias[o]))?;
                }
            }
            AnyModule::Activation(_) => {}
        }
    }
    Ok(())
}

pub fn export_data(data: &[Data]) -> Result<(), Box<dyn Error>> {
    let mut file_act = File::create("npu/data/activations.mem")?;
    let mut file_target = File::create("npu/data/target.mem")?;
    for sample in data {
        for input in &sample.input {
            writeln!(file_act, "{:08X}", to_q8_24(*input))?;
        }
        for target in &sample.target {
            writeln!(file_target, "{:08X}", to_q8_24(*target))?;
        }
    }
    Ok(())
}
