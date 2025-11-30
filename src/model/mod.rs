pub mod linear_layer {

    use ndarray::{Array1, Array2, LinalgScalar};

    pub trait Layer {
        fn forward(&mut self, prev_activation: &Array1<f32>);
        fn backward(&mut self, prev_activation: &Array1<f32>, next_derivative: &Array1<f32>);
        fn activation(&self) -> &Array1<f32>;
        fn prev_derivative(&self) -> &Array1<f32>;
    }

    pub struct LinearLayer {
        weights: Array2<f32>,
        biases: Array1<f32>,
    
        activation: Array1<f32>,
        prev_derivative: Array1<f32>,
    }
    
    pub struct SoftmaxLayer {
        activation: Array1<f32>,
        prev_derivative: Array1<f32>,
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