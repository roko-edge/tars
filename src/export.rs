use crate::{Data, modules::*, quantize_scalar};
use std::error::Error;
use std::fs::File;
use std::io::Write;

pub fn to_q8_24(x: f32) -> u32 {
    //WARNING: Rust conversion from f32 to u32 saturate in type limits
    ((x * 16_777_216.0) as i32) as u32
}
pub fn to_t3(x: f32, w0: f32) -> u8 {
    quantize_scalar(x, w0).bits() as u8
}

pub fn export_model(model: &Sequential, scales: &[f32]) -> Result<(), Box<dyn Error>> {
    let mut file_weights = File::create("npu/data/wt.mem")?;
    let mut file_scales = File::create("npu/data/w0.mem")?;
    let mut file_bias = File::create("npu/data/bias.mem")?;
    for &scale in scales {
        writeln!(file_scales, "{:08X}", to_q8_24(scale))?;
    }
    let mut pos = 0;
    for module in &model.modules {
        match module {
            AnyModule::Linear(linear) => {
                for o in 0..linear.out_sz {
                    for w in 0..linear.in_sz {
                        let code = to_t3(linear.weights.get(&[o, w]), scales[pos]);
                        writeln!(file_weights, "{:02X}", code)?;
                    }
                    writeln!(file_bias, "{:08X}", to_q8_24(linear.bias.get(&[o])))?;
                }
                pos += 1;
            }
            AnyModule::Activation(_) => {}
        }
    }
    Ok(())
}
pub fn export_inference(model: &Sequential, data: &[Data]) -> Result<(), Box<dyn Error>> {
    let mut file_input = File::create("npu/data/activations.mem")?;
    let mut file_target = File::create("npu/data/target.mem")?;
    for sample in data {
        for i in 0..sample.input.numel() {
            writeln!(file_input, "{:08X}", to_q8_24(sample.input.get(&[i])))?;
        }
        let output = model.forward(&sample.input);
        for o in 0..output.numel() {
            writeln!(file_target, "{:08X}", to_q8_24(output.get(&[o])))?;
        }
    }
    Ok(())
}
