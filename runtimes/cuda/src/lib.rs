//! CUDA Runtime Adapter.
//!
//! Adapter for CUDA-based inference backends (NVIDIA GPUs).

use cluster_types::{
    RuntimeBackend, TensorSpec, TensorHandle, ModelShardSpec, ShardHandle,
    StageExecution, StageOutput, TensorTransfer, TransferReceipt,
};
use observability::{LogContext, MetricsCollector};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};
use cudarc::driver::CudaDevice;
use cudarc::driver::result::DriverError;

/// CUDA runtime adapter.
pub struct CudaRuntime {
    backend: RuntimeBackend,
    metrics: MetricsCollector,
    _loaded_models: Vec<String>,
    device: Arc<RwLock<Option<Arc<CudaDevice>>>>,
    device_id: usize,
}

impl CudaRuntime {
    pub fn new(device_id: usize) -> Result<Self, CudaError> {
        info!("Initializing CUDA runtime on device {}", device_id);

        // Try to initialize CUDA device
        let device = match CudaDevice::new(device_id) {
            Ok(dev) => {
                info!("Successfully initialized CUDA device {}", device_id);
                Some(dev)
            }
            Err(e) => {
                warn!("Failed to initialize CUDA device {}: {}", device_id, e);
                None
            }
        };

        Ok(Self {
            backend: RuntimeBackend::CUDA,
            metrics: MetricsCollector::new("cuda-runtime".to_string()),
            _loaded_models: Vec::new(),
            device: Arc::new(RwLock::new(device)),
            device_id,
        })
    }

    /// Check if CUDA device is available.
    pub async fn is_available(&self) -> bool {
        let device = self.device.read().await;
        device.is_some()
    }

    /// Get device properties.
    pub async fn get_device_info(&self) -> Result<DeviceInfo, CudaError> {
        let device = self.device.read().await;
        
        if device.is_none() {
            return Err(CudaError::DeviceNotInitialized);
        }

        // Return simulated device info for compatibility
        Ok(DeviceInfo {
            name: format!("NVIDIA GPU {}", self.device_id),
            compute_capability: "8.0".to_string(),
            total_memory: 16 * 1024 * 1024 * 1024, // 16 GB
            multiprocessor_count: 28,
            max_threads_per_block: 1024,
            max_threads_per_multiprocessor: 2048,
            warp_size: 32,
        })
    }

    /// Probe runtime capabilities.
    pub fn probe(&self) -> RuntimeCapabilities {
        RuntimeCapabilities {
            backend: self.backend.clone(),
            supported_dtypes: vec![
                "f32".to_string(),
                "f16".to_string(),
                "bf16".to_string(),
                "i8".to_string(),
                "i4".to_string(),
            ],
            supported_precisions: vec![
                "fp32".to_string(),
                "fp16".to_string(),
                "bf16".to_string(),
                "int8".to_string(),
                "int4".to_string(),
            ],
            max_tensor_size: 16 * 1024 * 1024 * 1024, // 16 GB
            supports_flash_attention: true,
            supports_tensor_cores: true,
        }
    }

    /// Load a model shard.
    pub async fn load_shard(&self, spec: &ModelShardSpec) -> Result<ShardHandle, CudaError> {
        let ctx = LogContext::new("load_shard".to_string())
            .with_model_id(spec.model_id.clone());

        info!("Loading CUDA shard: {}", spec.shard_id);

        let device = self.device.read().await;
        
        if device.is_none() {
            return Err(CudaError::DeviceNotInitialized);
        }

        // In a real implementation, this would:
        // 1. Load the GGUF file
        // 2. Parse the model structure
        // 3. Allocate GPU memory for weights
        // 4. Load weights into GPU memory
        // 5. Initialize CUDA kernels

        let start = std::time::Instant::now();
        
        // Simulate loading time
        let load_time_ms = (spec.bytes / (100 * 1024 * 1024)) as u32; // Assume 100 MB/s
        tokio::time::sleep(std::time::Duration::from_millis(load_time_ms as u64)).await;

        let handle = ShardHandle {
            shard_id: spec.shard_id,
            backend: self.backend.clone(),
        };

        self.metrics.increment_counter("shards_loaded", 1, &[("device", "cuda")]);
        self.metrics.record_histogram("shard_load_ms", start.elapsed().as_secs_f64(), &[("device", "cuda")]);
        ctx.info(&format!("Shard {} loaded successfully in {}ms", spec.shard_id, load_time_ms));

        Ok(handle)
    }

    /// Allocate a tensor on GPU.
    pub async fn allocate_tensor(&self, spec: &TensorSpec) -> Result<TensorHandle, CudaError> {
        let ctx = LogContext::new("allocate_tensor".to_string())
            .with_session_id(spec.tensor_id.clone());

        info!("Allocating CUDA tensor: {} ({} bytes)", spec.tensor_id, spec.bytes);

        let device = self.device.read().await;
        
        if device.is_none() {
            return Err(CudaError::DeviceNotInitialized);
        }

        // Validate tensor specification
        if spec.bytes == 0 {
            return Err(CudaError::TensorAllocationFailed("Tensor size cannot be zero".to_string()));
        }

        // Check GPU memory availability
        let device_info = self.get_device_info().await?;
        if spec.bytes > device_info.total_memory {
            return Err(CudaError::TensorAllocationFailed(format!(
                "Tensor size {} exceeds GPU memory {}", spec.bytes, device_info.total_memory
            )));
        }

        let handle = TensorHandle {
            tensor_id: spec.tensor_id.clone(),
            device: format!("cuda:{}", self.device_id),
            offset: 0,
            size: spec.bytes,
        };

        self.metrics.increment_counter("tensors_allocated", 1, &[("device", "cuda")]);
        ctx.info(&format!("Tensor {} allocated on cuda:{}", spec.tensor_id, self.device_id));

        Ok(handle)
    }

    /// Execute a computation stage on GPU.
    pub async fn execute_stage(&self, request: StageExecution) -> Result<StageOutput, CudaError> {
        let ctx = LogContext::new("execute_stage".to_string());

        info!("Executing CUDA stage: {} with {} input tensors", request.stage_id, request.input_tensors.len());

        let device = self.device.read().await;
        
        if device.is_none() {
            return Err(CudaError::DeviceNotInitialized);
        }

        let start = std::time::Instant::now();
        
        // Simulate GPU computation time (faster than CPU)
        let total_input_bytes: u64 = request.input_tensors.iter().map(|t| t.size).sum();
        let compute_time_ms = (total_input_bytes / (500 * 1024 * 1024)) as u32; // Simulate 500 MB/s compute
        let compute_time_ms = compute_time_ms.max(1).min(500); // Clamp between 1ms and 500ms
        
        tokio::time::sleep(std::time::Duration::from_millis(compute_time_ms as u64)).await;

        let duration = start.elapsed();

        let output_tensors = vec![
            TensorHandle {
                tensor_id: format!("stage_{}_output_0", request.stage_id),
                device: format!("cuda:{}", self.device_id),
                offset: 0,
                size: total_input_bytes,
            }
        ];

        let output = StageOutput {
            stage_id: request.stage_id,
            output_tensors,
            execution_time_ms: duration.as_millis() as u32,
        };

        self.metrics.record_histogram("stage_execution_ms", duration.as_secs_f64(), &[("device", "cuda")]);
        ctx.info(&format!("Stage {} executed in {}ms on GPU", request.stage_id, duration.as_millis()));

        Ok(output)
    }

    /// Transfer a tensor between devices.
    pub async fn transfer_tensor(&self, request: TensorTransfer) -> Result<TransferReceipt, CudaError> {
        let ctx = LogContext::new("transfer_tensor".to_string())
            .with_session_id(request.tensor_id.clone());

        info!("Transferring tensor: {} from {} to {} ({} bytes)", 
            request.tensor_id, request.from_device, request.to_device, request.bytes);

        let start = std::time::Instant::now();
        
        // Simulate transfer time (GPU to CPU or GPU to GPU)
        let is_gpu_to_gpu = request.from_device.starts_with("cuda") && request.to_device.starts_with("cuda");
        let bandwidth_mbps = if is_gpu_to_gpu { 1000 } else { 50 }; // GPU-GPU is faster
        
        let transfer_time_ms = (request.bytes / (bandwidth_mbps * 1024 * 1024)) as u32;
        let transfer_time_ms = transfer_time_ms.max(1).min(2000);
        
        tokio::time::sleep(std::time::Duration::from_millis(transfer_time_ms as u64)).await;

        let duration = start.elapsed();

        let receipt = TransferReceipt {
            tensor_id: request.tensor_id.clone(),
            bytes_transferred: request.bytes,
            transfer_time_ms: duration.as_millis() as u32,
        };

        self.metrics.record_histogram("tensor_transfer_ms", duration.as_secs_f64(), &[("device", "cuda")]);
        self.metrics.increment_counter("tensor_bytes_transferred", request.bytes, &[("device", "cuda")]);
        ctx.info(&format!("Tensor {} transferred in {}ms", request.tensor_id, duration.as_millis()));

        Ok(receipt)
    }

    /// Unload a model shard.
    pub async fn unload(&self, handle: ShardHandle) -> Result<(), CudaError> {
        info!("Unloading CUDA shard: {}", handle.shard_id);

        self.metrics.increment_counter("shards_unloaded", 1, &[("device", "cuda")]);

        Ok(())
    }
}

/// Device information.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: String,
    pub total_memory: u64,
    pub multiprocessor_count: u32,
    pub max_threads_per_block: u32,
    pub max_threads_per_multiprocessor: u32,
    pub warp_size: u32,
}

/// Runtime capabilities.
#[derive(Debug, Clone)]
pub struct RuntimeCapabilities {
    pub backend: RuntimeBackend,
    pub supported_dtypes: Vec<String>,
    pub supported_precisions: Vec<String>,
    pub max_tensor_size: u64,
    pub supports_flash_attention: bool,
    pub supports_tensor_cores: bool,
}

/// CUDA runtime errors.
#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("Device not initialized")]
    DeviceNotInitialized,
    
    #[error("CUDA driver error: {0}")]
    DriverError(String),
    
    #[error("Model load failed: {0}")]
    ModelLoadFailed(String),
    
    #[error("Tensor allocation failed: {0}")]
    TensorAllocationFailed(String),
    
    #[error("Stage execution failed: {0}")]
    StageExecutionFailed(String),
    
    #[error("Tensor transfer failed: {0}")]
    TensorTransferFailed(String),
    
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
}

impl From<DriverError> for CudaError {
    fn from(e: DriverError) -> Self {
        CudaError::DriverError(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_runtime_creation() {
        let runtime = CudaRuntime::new(0);
        assert!(runtime.is_ok());
        
        if let Ok(runtime) = runtime {
            assert_eq!(runtime.backend, RuntimeBackend::CUDA);
            assert_eq!(runtime.device_id, 0);
        }
    }

    #[tokio::test]
    async fn test_device_info() {
        let runtime = CudaRuntime::new(0).unwrap();
        
        if runtime.is_available().await {
            let info = runtime.get_device_info().await;
            assert!(info.is_ok());
            
            if let Ok(info) = info {
                assert!(!info.name.is_empty());
                assert!(info.total_memory > 0);
            }
        }
    }
}
