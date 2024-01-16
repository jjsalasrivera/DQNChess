use rand::seq::SliceRandom;

#[derive(Clone)]
pub struct BufferElement {
    pub actual_state: String,
    pub action: usize,
    pub reward: f32,
    pub next_state: String,
    pub done: bool
}

pub struct ReplayBuffer {
    buffer: Vec<BufferElement>,
    capacity: usize,
    index: usize
}

impl ReplayBuffer{
    pub fn new(capacity: usize) -> ReplayBuffer{
        ReplayBuffer { buffer: Vec::with_capacity(capacity), capacity, index: 0}
    }

    pub fn add(&mut self, element: BufferElement) {
        if self.index >= self.buffer.len() {
            self.buffer.insert(self.index, element)
        }
        else {
            self.buffer[self.index] = element;
        }
        
        self.index += 1;
        
        if self.index == self.capacity {
            self.index = 0;
        }
    }

    pub fn get_samples(&self, samples: usize) -> Vec<BufferElement> {
        let mut rng = rand::thread_rng();
        self.buffer.choose_multiple(&mut rng, samples).cloned().collect()
    }
}