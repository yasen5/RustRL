use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

pub trait Layer {
    fn forward(&mut self, prev_activation: &Array1<f32>);
    fn backward(&mut self, prev_activation: &Array1<f32>, next_derivative: &Array1<f32>);
    fn activation(&self) -> &Array1<f32>;
    fn prev_derivative(&self) -> &Array1<f32>;
}

pub struct SoftmaxLayer {
    activation: Array1<f32>,
    prev_derivative: Array1<f32>,
}

impl SoftmaxLayer {
    pub fn new(inputs: usize) -> Self {
        Self {
            activation: Array1::zeros(inputs),
            prev_derivative: Array1::zeros(inputs),
        }
    }
}

impl Layer for SoftmaxLayer {
    fn forward(&mut self, prev_activation: &Array1<f32>) {
        let mut exp_sum: f32 = 0.;
        for num in prev_activation {
            exp_sum += num.exp();
        }
        for i in 0..prev_activation.len() {
            self.activation[i] = prev_activation[i].exp() / exp_sum;
        }
    }

    fn backward(&mut self, prev_activation: &Array1<f32>, next_derivative: &Array1<f32>) {
        // This assumes that the calculations are being done elsewhere, since it is
        // really convenient to calculate derivative if you do cross-entropy on a softmax
        self.prev_derivative = next_derivative.clone(); // TODO this is inefficient 
    }

    fn activation(&self) -> &Array1<f32> {
        &self.activation
    }

    fn prev_derivative(&self) -> &Array1<f32> {
        &self.prev_derivative
    }
}

pub struct LinearLayer {
    weights: Array2<f32>,
    biases: Array1<f32>,

    activation: Array1<f32>,
    prev_derivative: Array1<f32>,
}

impl LinearLayer {
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            weights: ndarray::Array2::random((outputs, inputs), Uniform::new(-1.0, 1.0).unwrap()),
            biases: ndarray::Array1::random(outputs, Uniform::new(-1.0, 1.0).unwrap()),
            activation: ndarray::Array1::zeros(outputs),
            prev_derivative: ndarray::Array1::zeros(inputs),
        }
    }
}

impl Layer for LinearLayer {
    fn forward(&mut self, prev_activation: &Array1<f32>) {
        self.activation = self.weights.dot(prev_activation) + &self.biases;
    }

    fn backward(&mut self, prev_activation: &Array1<f32>, next_derivative: &Array1<f32>) {
        self.weights -= &next_derivative
            .view()
            .insert_axis(ndarray::Axis(1))
            .dot(&prev_activation.view().insert_axis(ndarray::Axis(0)));
        self.biases -= next_derivative;
    }

    fn activation(&self) -> &Array1<f32> {
        &self.activation
    }

    fn prev_derivative(&self) -> &Array1<f32> {
        &self.prev_derivative
    }
}

pub struct Model {
    layers: Vec<Box<dyn Layer>>,
    num_layers: usize,
    action_space: usize,
}

impl Model {
    pub fn new(action_space: usize) -> Self {
        Self {
            layers: vec![],
            num_layers: 0,
            action_space: action_space,
        }
    }

    pub fn add_layer(&mut self, layer: Box<dyn Layer>) -> () {
        self.action_space = layer.activation().len();
        self.layers.push(layer);
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
        return &self.layers[self.layers.len() - 1].activation();
    }

    fn backprop(
        &mut self,
        state: Array1<f32>,
        choice: usize,
        true_q: f32,
        learning_rate: f32,
    ) -> f32 {
        let last_layer = &self.layers[self.layers.len() - 1];
        let mut loss_derivative: Array1<f32> = Array1::zeros(last_layer.activation().len());
        loss_derivative[choice] = last_layer.activation()[choice] - true_q;
        loss_derivative *= learning_rate;
        let (before_layers, last_layer) = self.layers.split_at_mut(self.num_layers);
        last_layer[0].backward(
            before_layers[before_layers.len() - 1].activation(),
            &loss_derivative,
        );
        for i in (0..self.layers.len() - 2).rev() {
            let (before_layers, after_layers) = self.layers.split_at_mut(i);
            let (curr_layer, after_layers) = after_layers.split_at_mut(1);
            curr_layer[0].backward(
                before_layers[before_layers.len() - 1].activation(),
                after_layers[0].prev_derivative(),
            );
        }
        return 2.;
    }
}
