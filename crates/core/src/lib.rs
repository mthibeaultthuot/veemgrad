#[derive(Debug)]

pub struct Tensor {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Tensor {
    pub fn new<const R: usize, const C: usize>(data: [[f32; C]; R]) -> Self {
        let flat: Vec<f32> = data.iter().flat_map(|row| row.iter()).copied().collect();
        Self {
            data: flat,
            rows: R,
            cols: C,
        }
    }

    pub fn new_increment(rows: usize, cols: usize, fill_fn: impl Fn(usize) -> f32) -> Self {
        let data = (0..rows * cols).map(fill_fn).collect();
        Tensor { data, rows, cols }
    }
}
