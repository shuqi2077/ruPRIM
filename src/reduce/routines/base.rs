use ruda_kernel::dsl as cubecl;
use crate::reduce::{ReduceDtypes, ReduceError, VectorizationMode, routines::ReduceBlueprint};
use ruda_kernel::dsl::prelude::*;

#[derive(Debug)]
pub struct ReduceVectorSettings {
    pub vectorization_mode: VectorizationMode,
    pub vector_size_input: VectorSize,
    pub vector_size_output: VectorSize,
}

#[derive(Debug)]
pub struct ReduceLaunchSettings {
    pub cube_dim: CubeDim,
    pub cube_count: CubeCount,
    pub address_type: AddressType,
    pub vector: ReduceVectorSettings,
}

#[derive(Debug)]
pub struct ReduceProblem {
    /// Number of elements in reduce axis
    pub reduce_len: usize,
    /// Number of instances of the reduce axis
    pub reduce_count: usize,
    pub axis: usize,
    pub dtypes: ReduceDtypes,
    /// The address type, defined by the max of each handle's `required_address_type`
    pub address_type: AddressType,
}

#[derive(Debug, Clone)]
pub enum BlueprintStrategy<R: Routine> {
    Forced(R::Blueprint, CubeDim),
    Inferred(R::Strategy),
}

pub trait Routine: core::fmt::Debug + Clone + Sized {
    type Strategy: core::fmt::Debug + Clone + Send + 'static;
    type Blueprint: core::fmt::Debug + Clone + Send + 'static;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceVectorSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError>;
}

pub(crate) fn validate_cube_dim<R: Runtime>(
    client: &ComputeClient<R>,
    cube_dim: CubeDim,
) -> Result<(), ReduceError> {
    let hardware = &client.properties().hardware;
    let units = cube_dim.x.checked_mul(cube_dim.y).and_then(|xy| xy.checked_mul(cube_dim.z));
    if cube_dim.x == 0 || cube_dim.y == 0 || cube_dim.z == 0 {
        return Err(ReduceError::Validation {
            details: "Cube dimensions must be nonzero",
        });
    }
    if cube_dim.x > hardware.max_cube_dim.0
        || cube_dim.y > hardware.max_cube_dim.1
        || cube_dim.z > hardware.max_cube_dim.2
        || units.is_none_or(|units| units > hardware.max_units_per_cube)
    {
        return Err(ReduceError::Validation {
            details: "Cube dimensions exceed device limits",
        });
    }
    Ok(())
}
