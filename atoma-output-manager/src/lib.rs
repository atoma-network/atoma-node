use atoma_types::Response;
use tokio::sync::mpsc;

pub struct AtomaOutputManager { 
    output_receiver: mpsc::Receiver<Response>,
}

impl AtomaOutputManager { 
    pub fn new(output_receiver: mpsc::Receiver<Response>) -> Self { 
        Self  {
            output_receiver
        }
    }
}