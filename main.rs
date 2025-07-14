use ndarray::{Array, Array1, Array2, Axis};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::f32::consts::E;
use ndarray::s;

#[derive(Debug)]
struct NeuralNetwork {
    input_size: usize,
    layer1_size: usize,
    layer2_size: usize,
    output_size: usize,
    learning_rate: f32,
    weights_input_to_layer1: Array2<f32>,           //dimensions: # of neurons x # inputs
    biases1: Array2<f32>,
    weights_layer1_to_layer2: Array2<f32>,
    biases2: Array2<f32>,
    weights_layer2_to_output: Array2<f32>,
    biases3: Array2<f32>,
}

// sigmoid activation function
fn sigmoid(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|x| { 1.0 / (1.0 + E.powf(-x)) } )
}

// Derivative of sigmoid
fn sigmoid_derivative(x: &Array2<f32>) -> Array2<f32> {
    x * &(1.0 - x)
}

fn softmax(x: &Array2<f32>) -> Array2<f32> {
    let max_per_col = x.map_axis(Axis(0), |col| col.fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    let shifted = x - &max_per_col.insert_axis(Axis(0));  //avoid overflow
    let exp_x = shifted.mapv(|x| x.exp());
    let sum_exp = exp_x.sum_axis(Axis(0)).insert_axis(Axis(0));
    &exp_x / &sum_exp
}

impl NeuralNetwork {
    fn new(input_size: usize, layer1_size: usize, layer2_size:usize, output_size: usize, learning_rate: f32) -> Self {
        let weights_input_to_layer1 = Array::random((layer1_size, input_size), Uniform::new(-0.5, 0.5));
        let weights_layer1_to_layer2 = Array::random((layer2_size, layer1_size), Uniform::new(-0.5, 0.5));
        let weights_layer2_to_output = Array::random((output_size, layer2_size), Uniform::new(-0.5, 0.5));

        let biases1 = Array2::zeros((layer1_size, 1));
        let biases2 = Array2::zeros((layer2_size, 1));
        let biases3 = Array2::zeros((output_size, 1));

        // Return a neural network that has the randomly initialized weights.
        NeuralNetwork {
            input_size,
            layer1_size,
            layer2_size,
            output_size,
            learning_rate,
            weights_input_to_layer1,
            biases1,
            weights_layer1_to_layer2,
            biases2,
            weights_layer2_to_output,
            biases3,
        }
    }

    // Forward propagation.  Returns all of the intermediate and final outputs.
    fn forward(&self, input: &Array2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) 
    {
        //hidden layer 1
        let pre_activation1 = self.weights_input_to_layer1.dot(input) + &self.biases1;
        let hidden_1_output = sigmoid(&pre_activation1);

        //hidden layer 2
        let pre_activation2 = self.weights_layer1_to_layer2.dot(&hidden_1_output) + &self.biases2;
        let hidden_2_output = sigmoid(&pre_activation2);

        // Output layer
        let pre_activation3 = self.weights_layer2_to_output.dot(&hidden_2_output) + &self.biases3;
        let final_output = softmax(&pre_activation3);

        (hidden_1_output, hidden_2_output, final_output)
    }

    // Backpropagation pass through the network. No return values but it is supposed to
    // update all weights.  It accepts the input, intermediate outputs and final outputs
    // as parameters as well as the target values.
    fn backward(
        &mut self,
        input: &Array2<f32>,
        layer1_output: &Array2<f32>,
        layer2_output: &Array2<f32>,
        final_output: &Array2<f32>,
        target: &Array2<f32>,
    ) {
        let batch_size = input.shape()[1] as f32;

        // Calculate gradients for output layer
        let output_error = final_output - target;
        
        // Gradients for weights_layer2_to_output and biases3
        let d_weights3 = output_error.dot(&layer2_output.t()) / batch_size;
        let d_biases3 = &output_error.sum_axis(Axis(1)).insert_axis(Axis(1)) / batch_size;

        // Backpropagate error to hidden layer 2
        let hidden_2_error = self.weights_layer2_to_output.t().dot(&output_error);
        let hidden_2_gradient = &hidden_2_error * &sigmoid_derivative(layer2_output);

        // Gradients for weights_layer1_to_layer2 and biases2
        let d_weights2 = hidden_2_gradient.dot(&layer1_output.t()) / batch_size; //shouldbe 397 397
        let d_biases2 = &hidden_2_gradient.sum_axis(Axis(1)).insert_axis(Axis(1)) / batch_size;

        // Backpropagate error to hidden layer 1
        let hidden_1_error = self.weights_layer1_to_layer2.t().dot(&hidden_2_error);
        let hidden_1_gradient = &hidden_1_error * &sigmoid_derivative(layer1_output);

        // Gradients for weights_input_to_layer1 and biases1
        let d_weights1 = hidden_1_gradient.dot(&input.t()) / batch_size;
        let d_biases1 = &hidden_1_gradient.sum_axis(Axis(1)).insert_axis(Axis(1)) / batch_size;

        // Update weights and biases using gradient descent
        self.weights_layer2_to_output = &self.weights_layer2_to_output - &(d_weights3 * self.learning_rate);
        self.biases3 = &self.biases3 - &(d_biases3 * self.learning_rate);
        self.weights_layer1_to_layer2 = &self.weights_layer1_to_layer2 - &(d_weights2 * self.learning_rate); 
        self.biases2 = &self.biases2 - &(d_biases2 * self.learning_rate);
        self.weights_input_to_layer1 = &self.weights_input_to_layer1 - &(d_weights1 * self.learning_rate);
        self.biases1 = &self.biases1 - &(d_biases1 * self.learning_rate);
    }

    fn train(&mut self, input: &Array2<f32>, target: &Array2<f32>) -> f32 {
        // Forward pass
        let (hidden_1_output, hidden_2_output, final_output) = self.forward(input);

        // Calculate loss (cross-entropy)
        let epsilon = 1e-15;
        let loss = -target * &final_output.mapv(|x| (x + epsilon).ln());
        let loss = loss.sum() / (input.shape()[1] as f32);

        // Backward pass
        self.backward(input, &hidden_1_output, &hidden_2_output, &final_output, target);

        loss
    }    

    fn read_csv(&mut self, path: &str) -> (Array1<f32>, Array2<f32>) {
        let mut rdr = csv::ReaderBuilder::new()
            .delimiter(b',')
            .has_headers(false)
            .flexible(true)
            .from_path(path)
            .expect("can't read csv");
    
        let num_features = 784;
        let mut labels: Vec<f32> = Vec::new();
        let mut flat_data: Vec<f32> = Vec::new();

        for result in rdr.records() {  //FOR ROW IN THE CSV
            let record = result.unwrap();
            let row: Vec<f32> = record.iter().map(|x| x.parse::<f32>().expect("Failed to parse")).collect();
            let label = row[0];
            labels.push(label);
            flat_data.extend(row[1..].iter().map(|&x| x as f32 / 255.0)); //normalized
        }

        let num_samples = labels.len();
        let features_matrix = Array2::from_shape_vec((num_samples, num_features), flat_data).expect("Shape mismatch");
        let features = features_matrix.t().to_owned();
        
        (Array1::from(labels), features)
    }

   
}

fn main() {

    let mut nn = NeuralNetwork::new(784, 512, 512, 10, 0.3);
    let train_path = "mnist_train.csv";
    let (train_labels, train_features) = nn.read_csv(train_path);
    let num_samples = 60000;

    // Create sample input
    let input = train_features;

    // Create sample target
    let mut target = Array2::<f32>::zeros((10, num_samples));       //one hot encoding
    for (i, &label) in train_labels.iter().enumerate() {
        target[[label as usize, i]] = 1.0;
        }    

    fn compute_loss(output: &Array2<f32>, target: &Array2<f32>) -> f32 {
        let epsilon = 1e-15;
        let log_preds = output.mapv(|x| (x + epsilon).ln());
        let loss = -target * &log_preds;
        loss.sum() / (target.shape()[1] as f32)
        }
     
    // Calculate initial cross entropy loss before training
    let (_, _, initial_output) = nn.forward(&input);
    println!("Initial loss before training: {}", compute_loss(&initial_output, &target));

    for epoch in 1..=3 {
        for start in (0..num_samples).step_by(250) {
            let end = (start + 250).min(num_samples);
            let input_batch = input.slice(s![.., start..end]).to_owned();
            let target_batch = target.slice(s![.., start..end]).to_owned();
            let loss = nn.train(&input_batch, &target_batch);
        }
        let (_, _, output) = nn.forward(&input);
        let epoch_loss = compute_loss(&output, &target);
        println!("Epoch {}: Loss = {}", epoch, epoch_loss);
    }

    let test_path = "mnist_test.csv";
    let (test_labels, test_features) = nn.read_csv(test_path);

    let (_, _, test_output) = nn.forward(&test_features);

    fn compute_accuracy(output: &Array2<f32>, labels: &Array1<f32>) -> f32 {
        let predictions = output.map_axis(Axis(0), |col| col.iter().cloned().enumerate().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap().0 as u32);
        let labels_int = labels.map(|x| *x as u32);
        let correct = predictions.iter().zip(labels_int.iter()).filter(|(p, l)| p == l).count();
        correct as f32 / labels.len() as f32
    }

    let test_accuracy = compute_accuracy(&test_output, &test_labels);
    println!("Test Accuracy: {:.2}%", test_accuracy * 100.0);

}


