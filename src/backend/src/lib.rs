use candid::{CandidType, Deserialize};
use ic_stable_structures::{
    memory_manager::{MemoryId, MemoryManager},
    DefaultMemoryImpl,
};
use std::cell::RefCell;

mod onnx;

// WASI polyfill requires a virtual stable memory to store the file system.
const WASI_MEMORY_ID: MemoryId = MemoryId::new(0);

thread_local! {
    static MEMORY_MANAGER: RefCell<MemoryManager<DefaultMemoryImpl>> =
        RefCell::new(MemoryManager::init(DefaultMemoryImpl::default()));
}

#[derive(CandidType, Deserialize)]
struct Prediction {
    values: Vec<f32>,
}

#[derive(CandidType, Deserialize)]
struct PredictionError {
    message: String,
}

#[derive(CandidType, Deserialize)]
enum PredictionResult {
    Ok(Prediction),
    Err(PredictionError),
}

/// Predict the future sequence using the LSTM/GRU model.
#[ic_cdk::update]
fn predict(future_steps: usize) -> PredictionResult {
    let result = match onnx::predict_sequence(future_steps) {
        Ok(predictions) => PredictionResult::Ok(Prediction {
            values: predictions,
        }),
        Err(err) => PredictionResult::Err(PredictionError {
            message: err.to_string(),
        }),
    };
    result
}

#[ic_cdk::init]
fn init() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
    onnx::setup().unwrap();
}

#[ic_cdk::post_upgrade]
fn post_upgrade() {
    let wasi_memory = MEMORY_MANAGER.with(|m| m.borrow().get(WASI_MEMORY_ID));
    ic_wasi_polyfill::init_with_memory(&[0u8; 32], &[], wasi_memory);
    onnx::setup().unwrap();
}
