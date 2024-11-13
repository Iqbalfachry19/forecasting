use ndarray::{s, Array2};
use prost::Message;

use std::cell::RefCell;
use tract_onnx::prelude::*;
type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

thread_local! {
    static MODEL: RefCell<Option<Model>> = RefCell::new(None);
}

/// The serialized ONNX model for time-series prediction.
const LSTM_GRU_MODEL: &'static [u8] = include_bytes!("../assets/lstmgru.onnx");

/// The expected input length for the model.
const EXPECTED_LENGTH: usize = 60;
const SCALE_FACTOR: f32 = 73440.0;
/// Constructs a runnable model from the serialized ONNX model.
pub fn setup() -> TractResult<()> {
    let bytes = bytes::Bytes::from_static(LSTM_GRU_MODEL);
    let proto: tract_onnx::pb::ModelProto = tract_onnx::pb::ModelProto::decode(bytes)?;
    let model = tract_onnx::onnx()
        .model_for_proto_model(&proto)?
        .into_optimized()?
        .into_runnable()?;
    MODEL.with(|m| {
        *m.borrow_mut() = Some(model);
    });
    Ok(())
}

/// Runs the model on the given input sequence and returns the predictions.
pub fn predict_sequence(future_steps: usize) -> Result<Vec<f32>, anyhow::Error> {
    MODEL.with(|model| {
        let model = model.borrow();
        let model = model
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not initialized"))?;
        let last_sequence = vec![
            vec![0.8321338, 0.04469836, 0.67523277],
            vec![0.8013434, 0.06468637, 0.6197248],
            vec![0.81012595, 0.10549171, 0.689262],
            vec![0.8274835, 0.08625317, 0.70400137],
            vec![0.8015893, 0.08521014, 0.62818354],
            vec![0.78530884, 0.10151194, 0.6443225],
            vec![0.8037584, 0.08346964, 0.7068815],
            vec![0.8118472, 0.03855389, 0.68799394],
            vec![0.7980861, 0.05038306, 0.6492143],
            vec![0.8120475, 0.07366724, 0.69847715],
            vec![0.8053999, 0.0899171, 0.6619102],
            vec![0.83530617, 0.09310243, 0.72583646],
            vec![0.82433504, 0.07855322, 0.65474904],
            vec![0.8756787, 0.12102821, 0.76016164],
            vec![0.87684923, 0.06089855, 0.67558223],
            vec![0.8789867, 0.05348092, 0.6771559],
            vec![0.8588931, 0.07871369, 0.640557],
            vec![0.8121952, 0.11126316, 0.59366214],
            vec![0.805605, 0.11464206, 0.66201425],
            vec![0.8105915, 0.09165999, 0.6824968],
            vec![0.80687535, 0.0918531, 0.6670957],
            vec![0.80480665, 0.03517347, 0.66999704],
            vec![0.7820642, 0.0699091, 0.63267225],
            vec![0.80677855, 0.07687392, 0.71816707],
            vec![0.78352374, 0.07582095, 0.6318378],
            vec![0.7909992, 0.10135683, 0.6872484],
            vec![0.76595205, 0.08825535, 0.62766105],
            vec![0.7353634, 0.14049543, 0.6154263],
            vec![0.738004, 0.0541472, 0.67879057],
            vec![0.7477111, 0.05188677, 0.6923413],
            vec![0.7778328, 0.09847976, 0.7301244],
            vec![0.78653437, 0.08206382, 0.68957704],
            vec![0.78230876, 0.10540742, 0.665966],
            vec![0.79314935, 0.09625014, 0.69334996],
            vec![0.8269543, 0.0924167, 0.73337907],
            vec![0.8191239, 0.04664356, 0.66005576],
            vec![0.8077516, 0.05146693, 0.65366733],
            vec![0.7940552, 0.09111235, 0.6492075],
            vec![0.82332027, 0.10833272, 0.7254445],
            vec![0.84186846, 0.11664014, 0.7055521],
            vec![0.8597201, 0.12154043, 0.70370644],
            vec![0.86321247, 0.10007297, 0.67947346],
            vec![0.8660043, 0.04088765, 0.67829245],
            vec![0.86951536, 0.05734423, 0.67946273],
            vec![0.8651048, 0.08930977, 0.6663879],
            vec![0.87855, 0.08514357, 0.69575274],
            vec![0.8625233, 0.07129388, 0.64730704],
            vec![0.89070743, 0.10490605, 0.71971416],
            vec![0.8991389, 0.09118642, 0.6871645],
            vec![0.9004802, 0.04326725, 0.6758046],
            vec![0.89699024, 0.04196941, 0.66810685],
            vec![0.8651006, 0.1055895, 0.62182826],
            vec![0.83062905, 0.14294402, 0.61546385],
            vec![0.8278046, 0.11599045, 0.668795],
            vec![0.82955575, 0.10272119, 0.67669326],
            vec![0.84764665, 0.084138, 0.70454603],
            vec![0.84795743, 0.03774378, 0.67419404],
            vec![0.8580397, 0.04193527, 0.69058996],
            vec![0.8499865, 0.09744092, 0.6601698],
            vec![0.8485386, 0.08000301, 0.6712288],
        ];
        // Ensure last_sequence has expected shape
        let input_length = last_sequence.len();
        let depth = if !last_sequence.is_empty() {
            last_sequence[0].len()
        } else {
            return Err(anyhow::anyhow!("Input sequence is empty."));
        };

        // Validate shape
        if input_length != EXPECTED_LENGTH || depth != 3 {
            return Err(anyhow::anyhow!(
                "Input sequence shape does not match expected shape of (60, 3). Got ({}, {}).",
                input_length,
                depth
            ));
        }

        // Convert to Array2<f32>
        let mut current_sequence = Array2::from_shape_vec(
            (input_length, depth),
            last_sequence.into_iter().flatten().collect(),
        )?;

        let mut predictions = Vec::new();

        for _ in 0..future_steps {
            println!("Current sequence shape: {:?}", current_sequence.shape());

            let input_tensor =
                Tensor::from(current_sequence.clone().insert_axis(ndarray::Axis(0))).into();
            let result = model.run(tvec!(input_tensor))?;

            let next_pred = result[0].to_array_view::<f32>()?;
            println!("Next prediction shape: {:?}", next_pred.shape());

            let next_value = next_pred[[0, 0]];

            // Inverse transform the prediction
            let next_value_inv = next_value * SCALE_FACTOR;
            predictions.push(next_value_inv);

            println!("Predicted value: {}", next_value_inv);

            let next_pred_reshaped = Array2::from_elem((1, depth), next_value);
            println!("next_pred_reshaped shape: {:?}", next_pred_reshaped.shape());

            current_sequence = ndarray::concatenate![
                ndarray::Axis(0),
                current_sequence.slice(s![1.., ..]),
                next_pred_reshaped
            ];
            println!(
                "Updated current_sequence shape: {:?}",
                current_sequence.shape()
            );
        }

        Ok(predictions)
    })
}
