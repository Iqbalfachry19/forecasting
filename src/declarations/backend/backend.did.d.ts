import type { Principal } from '@dfinity/principal';
import type { ActorMethod } from '@dfinity/agent';
import type { IDL } from '@dfinity/candid';

export interface Prediction { 'values' : Array<number> }
export interface PredictionError { 'message' : string }
export type PredictionResult = { 'Ok' : Prediction } |
  { 'Err' : PredictionError };
export interface _SERVICE {
  'predict' : ActorMethod<[bigint], PredictionResult>,
}
export declare const idlFactory: IDL.InterfaceFactory;
export declare const init: (args: { IDL: typeof IDL }) => IDL.Type[];
