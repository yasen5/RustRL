use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub struct SoftmaxLayer {
    activation: Array1<f32>,
}

impl SoftmaxLayer {
    pub fn new(inputs: usize) -> Self {
        Self {
            activation: Array1::zeros(inputs),
        }
    }
}

impl SoftmaxLayer {
    fn forward(&mut self, prev_activation: &Array1<f32>) {
        let mut exp_sum: f32 = 0.;
        for num in prev_activation {
            exp_sum += num.exp();
        }
        for i in 0..prev_activation.len() {
            self.activation[i] = prev_activation[i].exp() / exp_sum;
        }
    }

    fn activation(&self) -> &Array1<f32> {
        &self.activation
    }
}

#[derive(Clone)]
pub struct LinearLayer {
    weights: Array2<f32>,
    weight_gradient: Array2<f32>,
    biases: Array1<f32>,
    bias_gradient: Array1<f32>,
    activation: Array1<f32>,
    prev_derivative: Array1<f32>,
}

impl LinearLayer {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            weights: ndarray::Array2::random((outputs, inputs), Uniform::new(-1.0, 1.0).unwrap()),
            weight_gradient: Array2::zeros((outputs, inputs)),
            biases: ndarray::Array1::random(outputs, Uniform::new(-1.0, 1.0).unwrap()),
            bias_gradient: Array1::zeros(outputs),
            activation: ndarray::Array1::zeros(outputs),
            prev_derivative: ndarray::Array1::zeros(inputs),
        }
    }
}

impl LinearLayer {
    fn forward(&mut self, prev_activation: &Array1<f32>) {
        self.activation =
            (self.weights.dot(prev_activation) + &self.biases).mapv(|num: f32| f32::max(0., num));
    }

    fn compute_gradient(&mut self, prev_activation: &Array1<f32>, next_derivative: &Array1<f32>) {
        self.prev_derivative = self
            .weights
            .dot(&(prev_activation.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 }) * next_derivative));
        self.weight_gradient -= &next_derivative
            .view()
            .insert_axis(ndarray::Axis(1))
            .dot(&prev_activation.view().insert_axis(ndarray::Axis(0)));
        self.bias_gradient -= next_derivative;
    }

    fn apply_gradient(&mut self, learning_rate: f32) {
        self.weight_gradient *= learning_rate;
        self.bias_gradient *= learning_rate;
        self.weights += &self.weight_gradient;
        self.biases += &self.bias_gradient;
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
    action_space: usize,
    learning_rate: f32,
}

impl Model {
    pub fn new(action_space: usize) -> Self {
        Self {
            layers: vec![],
            num_layers: 0,
            action_space: action_space,
            learning_rate: 0.001,
        }
    }

    pub fn zero_gradients(&mut self) {
        for layer in &mut self.layers {
            layer.zero_gradient();
        }
    }

    pub fn add_layer(&mut self, input_size: usize, output_size: usize) -> () {
        self.action_space = output_size;
        self.layers.push(LinearLayer::new(input_size, output_size));
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

    pub fn backprop(&mut self, state: Array1<f32>, loss_derivative: &mut Array1<f32>) {
        *loss_derivative *= self.learning_rate;
        let (before_layers, last_layer) = self.layers.split_at_mut(self.num_layers - 1);
        last_layer[0].compute_gradient(
            before_layers[before_layers.len() - 1].activation(),
            loss_derivative,
        );
        for i in (1..self.layers.len() - 2).rev() {
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
}
