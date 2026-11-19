use mnist_reader::{MnistReader, print_image};

fn main() {
    let mut mnist = MnistReader::new("mnist-data");
    mnist.load().unwrap();

    let train_data: Vec<Vec<f32>> = mnist.train_data;
    println!("images[0]={:?}", train_data[0]);
    print_image(&train_data[0]);

    let train_labels: Vec<u8> = mnist.train_labels;
    println!("labels[0]={:?}", train_labels[0]);
}
