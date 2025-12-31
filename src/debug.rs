#[cfg(debug_assertions)]
pub trait NdDebug<T> {
    fn dbg_vec(&self) -> Vec<T>;
}

#[cfg(debug_assertions)]
impl<T: Copy> NdDebug<T> for ndarray::Array1<T> {
    fn dbg_vec(&self) -> Vec<T> {
        self.to_vec()
    }
}