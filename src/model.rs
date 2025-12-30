use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

#[derive(Clone)]
pub struct LinearLayer {
    weights: Array2<f32>,
    weight_gradient: Array2<f32>,
    biases: Array1<f32>,
    bias_gradient: Array1<f32>,
    activation: Array1<f32>,
    prev_derivative: Array1<f32>,
    relu: bool,
}

impl LinearLayer {
    pub fn new(inputs: usize, outputs: usize, relu: bool) -> Self {
        Self {
            weights: ndarray::Array2::random((outputs, inputs), Uniform::new(-0.5, 1.0).unwrap())
                / 100.,
            weight_gradient: Array2::zeros((outputs, inputs)),
            biases: ndarray::Array1::random(outputs, Uniform::new(0., 1.0).unwrap()),
            bias_gradient: Array1::zeros(outputs),
            activation: ndarray::Array1::zeros(outputs),
            prev_derivative: ndarray::Array1::zeros(inputs),
            relu: relu,
        }
    }
}

impl LinearLayer {
    fn forward(&mut self, prev_activation: &Array1<f32>) {
        self.activation = self.weights.dot(prev_activation) + &self.biases;
        if self.relu {
            self.activation.mapv_inplace(|x| x.max(0.));
        }
    }

    fn compute_gradient(&mut self, prev_activation: &Array1<f32>, next_derivative: &Array1<f32>) {
        let relu_derivative: Array1<f32>;
        if self.relu {
            relu_derivative =
                self.activation.mapv(|x| if x >= 0.0 { 1.0 } else { 0.0 }) * next_derivative;
            self.prev_derivative = relu_derivative.dot(&self.weights);
            self.weight_gradient += &relu_derivative
                .view()
                .insert_axis(ndarray::Axis(1))
                .dot(&prev_activation.view().insert_axis(ndarray::Axis(0)));
            self.bias_gradient += &relu_derivative;
        }
        self.prev_derivative = next_derivative.dot(&self.weights);
        self.weight_gradient += &next_derivative
            .view()
            .insert_axis(ndarray::Axis(1))
            .dot(&prev_activation.view().insert_axis(ndarray::Axis(0)));
        self.bias_gradient += next_derivative;
    }

    fn apply_gradient(&mut self, learning_rate: f32) {
        self.weight_gradient *= learning_rate;
        self.bias_gradient *= learning_rate;
        self.weights -= &self.weight_gradient;
        self.biases -= &self.bias_gradient;
        self.zero_gradient();
    }

    fn zero_gradient(&mut self) {
        self.weight_gradient.fill(0.);
        self.bias_gradient.fill(0.);
    }

    fn activation(&self) -> &Array1<f32> {
        &self.activation
    }

    fn prev_derivative(&self) -> &Array1<f32> {
        &self.prev_derivative
    }
}

#[derive(Clone)]
pub struct Model {
    layers: Vec<LinearLayer>,
    num_layers: usize,
    learning_rate: f32,
}

impl Model {
    pub fn new() -> Self {
        Self {
            layers: vec![],
            num_layers: 0,
            learning_rate: 0.001,
        }
    }

    pub fn add_layer(&mut self, input_size: usize, output_size: usize, relu: bool) -> () {
        self.layers
            .push(LinearLayer::new(input_size, output_size, relu));
        self.num_layers += 1;
    }

    pub fn forward(&mut self, state: &Array1<f32>) -> &Array1<f32> {
        self.layers[0].forward(&state);
        for i in 1..self.layers.len() {
            let (prev_layers, next_layers) = self.layers.split_at_mut(i);
            let prev = prev_layers.last().unwrap();
            let curr = &mut next_layers[0];
            curr.forward(prev.activation());
        }
        &self.layers[self.layers.len() - 1].activation()
    }

    pub fn backprop(&mut self, state: &Array1<f32>, loss_derivative: &mut Array1<f32>) {
        let (before_layers, last_layer) = self.layers.split_at_mut(self.num_layers - 1);
        last_layer[0].compute_gradient(
            before_layers[before_layers.len() - 1].activation(),
            loss_derivative,
        );
        for i in (1..self.layers.len() - 1).rev() {
            let (before_layers, after_layers) = self.layers.split_at_mut(i);
            let (curr_layer, after_layers) = after_layers.split_at_mut(1);
            curr_layer[0].compute_gradient(
                before_layers[before_layers.len() - 1].activation(),
                after_layers[0].prev_derivative(),
            );
        }
        let (first_layer, other_layers) = self.layers.split_at_mut(1);
        first_layer[0].compute_gradient(&state, other_layers[0].prev_derivative());
    }

    pub fn apply_gradients(&mut self) {
        for layer in &mut self.layers {
            layer.apply_gradient(self.learning_rate);
        }
    }
}
