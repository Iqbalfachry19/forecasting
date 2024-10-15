export const idlFactory = ({ IDL }) => {
  const Prediction = IDL.Record({ 'values' : IDL.Vec(IDL.Float32) });
  const PredictionError = IDL.Record({ 'message' : IDL.Text });
  const PredictionResult = IDL.Variant({
    'Ok' : Prediction,
    'Err' : PredictionError,
  });
  return IDL.Service({
    'predict' : IDL.Func([IDL.Nat64], [PredictionResult], []),
  });
};
export const init = ({ IDL }) => { return []; };
