#![cfg(feature = "kernel-ir")]

use ruda_kernel::dsl as kernel_dsl;
use ruda_kernel::dsl::prelude::*;
use ruprim::reduce::components::instructions::{ArgMax, Item, ReduceInstruction, ReduceStep, Value};
use ruprim::reduce::components::precision::S;

#[ruda]
fn scalar_accumulator_updates(value: u32) -> u32 {
    let mut accumulator = Value::<u32>::new_single(0u32);
    accumulator.assign(&Value::<u32>::new_single(value));
    accumulator.assign(&Value::<u32>::new_single(accumulator.item() + 1u32));
    accumulator.item()
}

#[ruda]
fn argmax_accumulator_updates(value: Vector<f32, S>) -> Vector<u32, S> {
    let instruction = ArgMax {};
    let mut accumulator = <ArgMax as ReduceInstruction<f32>>::null_accumulator(&instruction);
    <ArgMax as ReduceInstruction<f32>>::reduce(&instruction, &mut accumulator,
        Item::<f32> { elements: value, args: Value::new_single(Vector::new(7u32)) },
        ReduceStep::Identity);
    <ArgMax as ReduceInstruction<f32>>::reduce(&instruction, &mut accumulator,
        Item::<f32> { elements: value, args: Value::new_single(Vector::new(3u32)) },
        ReduceStep::Identity);
    accumulator.args.item()
}

#[test]
fn scalar_accumulator_is_writable_after_constant_initialization() {
    let mut scope = Scope::root(false);
    scalar_accumulator_updates::expand(&mut scope, 9u32.into());
    assert!(!scope.instructions.is_empty());
}

#[test]
fn argmax_accumulator_expands_scalar_and_vector_updates() {
    for width in [1, 2, 4] {
        let mut scope = Scope::root(false);
        scope.register_size::<S>(width);
        let value = Vector::<f32, S>::__expand_new(&mut scope, 2.0f32.into());
        argmax_accumulator_updates::expand(&mut scope, value);
        assert!(!scope.instructions.is_empty());
    }
}
