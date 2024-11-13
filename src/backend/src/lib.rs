use candid::{CandidType, Deserialize};
use ic_cdk::api::management_canister::http_request::{
    http_request, CanisterHttpRequestArgument, HttpMethod,
};
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
#[ic_cdk::update]
async fn fetch_historical_price() -> PredictionResult {
    // Set up the CoinGecko API request URL
    let url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=30";

    // Configure the request to GET historical price data
    let request = CanisterHttpRequestArgument {
        url: url.to_string(),
        method: HttpMethod::GET,
        body: None,
        max_response_bytes: None,
        transform: None,
        headers: vec![],
    };
    let cycles: u128 = 230_949_972_000; // Example cycle amount; adjust as needed.

    // Send the HTTP request
    match http_request(request, cycles).await {
        Ok((response,)) => {
            // Parse the response body to extract the price data
            match String::from_utf8(response.body) {
                Ok(body_str) => {
                    // Convert JSON response to a structured format
                    let prices: Result<Vec<f32>, _> = parse_historical_prices(&body_str);
                    match prices {
                        Ok(values) => PredictionResult::Ok(Prediction { values }),
                        Err(err) => PredictionResult::Err(PredictionError {
                            message: format!("Failed to parse prices: {:?}", err),
                        }),
                    }
                }
                Err(_) => PredictionResult::Err(PredictionError {
                    message: "Response body is not UTF-8 encoded.".to_string(),
                }),
            }
        }
        Err((r, m)) => PredictionResult::Err(PredictionError {
            message: format!("HTTP request error: {:?} {}", r, m),
        }),
    }
}

/// Helper function to parse the JSON response and extract prices.
fn parse_historical_prices(response: &str) -> Result<Vec<f32>, serde_json::Error> {
    #[derive(Deserialize)]
    struct MarketChart {
        prices: Vec<(f64, f64)>, // (timestamp, price) tuples
    }

    let market_chart: MarketChart = serde_json::from_str(response)?;
    // Extract just the prices as f32 values
    let prices = market_chart
        .prices
        .into_iter()
        .map(|(_, price)| price as f32)
        .collect();
    Ok(prices)
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
