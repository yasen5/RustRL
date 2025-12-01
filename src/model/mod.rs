pub mod linear_layer {

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
        fn new(inputs: usize) -> Self {
            Self {
                activation: Array1::zeros(inputs),
                prev_derivative: Array1::zeros(inputs)
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
            // This assumes that 
            self.prev_derivative = next_derivative.clone(); // TODO is this inefficient?
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
        fn new(inputs: usize, outputs: usize) -> Self {
            Self {
                weights: ndarray::Array2::random((outputs, inputs), Uniform::new(-1.0, 1.0).unwrap()),
                biases: ndarray::Array1::random(outputs, Uniform::new(-1.0, 1.0).unwrap()),
                activation: ndarray::Array1::zeros(outputs),
                prev_derivative: ndarray::Array1::zeros(inputs)
            }
        }
    }

    impl Layer for LinearLayer {
        fn forward(&mut self, prev_activation: &Array1<f32>) {
            self.activation = self.weights.dot(prev_activation) + &self.biases;
        }

        fn backward(&mut self, prev_activation: &Array1<f32>, next_derivative: &Array1<f32>) {
            self.weights -= &next_derivative.view().insert_axis(ndarray::Axis(1)).dot(&prev_activation.view().insert_axis(ndarray::Axis(0)));
            self.biases -= next_derivative;
        }

        fn activation(&self) -> &Array1<f32> {
            &self.activation
        }

        fn prev_derivative(&self) -> &Array1<f32> {
            &self.prev_derivative
        }
    }
}