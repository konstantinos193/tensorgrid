//! Hardware detection and capability probing.
//!
//! This crate provides cross-platform hardware detection for CPU, GPU,
//! memory, storage, and network devices.

use cluster_types::{
    CpuCapabilities, GpuCapabilities, GpuVendor, MemoryCapabilities,
    NetworkCapabilities, NetworkInterface, NodeCapabilities, Precision,
    RuntimeBackend, StorageCapabilities,
};
use anyhow::Result;
use std::collections::HashMap;

/// Hardware probe that detects system capabilities.
pub struct HardwareProbe;

impl HardwareProbe {
    /// Probe all hardware capabilities of the current system.
    pub fn probe_all() -> Result<NodeCapabilities> {
        Ok(NodeCapabilities {
            cpu: Self::probe_cpu()?,
            gpus: Self::probe_gpus()?,
            memory: Self::probe_memory()?,
            storage: Self::probe_storage()?,
            network: Self::probe_network()?,
            supported_runtimes: Self::detect_supported_runtimes(),
        })
    }

    /// Probe CPU capabilities.
    pub fn probe_cpu() -> Result<CpuCapabilities> {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_all();

        let cpu = sys.cpus();
        let cores = sys.physical_core_count().unwrap_or(1) as u32;
        let threads = cpu.len() as u32;
        
        // Get CPU frequency (approximate from first CPU)
        let frequency_mhz = cpu.first()
            .and_then(|c| c.frequency())
            .unwrap_or(0);

        // Detect CPU architecture
        let architecture = std::env::consts::ARCH.to_string();

        // Detect CPU features (basic set)
        let features = vec![
            "aes".to_string(),
            if cfg!(target_arch = "x86_64") { "avx2".to_string() } else { String::new() },
            if cfg!(target_arch = "x86_64") { "sse4.2".to_string() } else { String::new() },
        ].into_iter().filter(|s| !s.is_empty()).collect();

        // Detect NUMA nodes (simplified - assume 1 for now)
        let numa_nodes = 1;

        Ok(CpuCapabilities {
            architecture,
            cores,
            threads,
            frequency_mhz,
            features,
            numa_nodes,
        })
    }

    /// Probe GPU capabilities.
    pub fn probe_gpus() -> Result<Vec<GpuCapabilities>> {
        let mut gpus = Vec::new();

        // Try to detect NVIDIA GPUs
        #[cfg(feature = "nvidia")]
        {
            if let Ok(nvidia_gpus) = Self::probe_nvidia_gpus() {
                gpus.extend(nvidia_gpus);
            }
        }

        // Fallback: on Windows, use WMI
        #[cfg(target_os = "windows")]
        {
            if let Ok(wmi_gpus) = Self::probe_wmi_gpus() {
                gpus.extend(wmi_gpus);
            }
        }

        // If no GPUs detected, return empty list
        Ok(gpus)
    }

    #[cfg(feature = "nvidia")]
    fn probe_nvidia_gpus() -> Result<Vec<GpuCapabilities>> {
        // Placeholder for NVIDIA NVML integration
        // In a real implementation, this would use the NVML library
        Ok(vec![])
    }

    #[cfg(target_os = "windows")]
    fn probe_wmi_gpus() -> Result<Vec<GpuCapabilities>> {
        use wmi::{COMLibrary, WMIConnection};
        
        let com = COMLibrary::new()?;
        let wmi = WMIConnection::new(com)?;
        
        let results: Vec<HashMap<String, String>> = wmi.query()?;
        
        let mut gpus = Vec::new();
        for (i, gpu_info) in results.iter().enumerate() {
            let name = gpu_info.get("Name").unwrap_or(&"Unknown GPU".to_string()).clone();
            let vendor_str = gpu_info.get("AdapterCompatibility").unwrap_or(&"Unknown".to_string()).clone();
            
            let vendor = if vendor_str.contains("NVIDIA") {
                GpuVendor::Nvidia
            } else if vendor_str.contains("AMD") || vendor_str.contains("ATI") {
                GpuVendor::Amd
            } else if vendor_str.contains("Intel") {
                GpuVendor::Intel
            } else {
                GpuVendor::Unknown
            };

            // Estimate VRAM from WMI data
            let vram_bytes = gpu_info.get("AdapterRAM")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);

            gpus.push(GpuCapabilities {
                id: format!("gpu:{}", i),
                name,
                vendor,
                vram_bytes,
                compute_capability: None,
                supports_peer_to_peer: false,
                supports_nvml: vendor == GpuVendor::Nvidia,
                supported_precisions: vec![Precision::Fp16, Precision::Fp32],
            });
        }
        
        Ok(gpus)
    }

    /// Probe memory capabilities.
    pub fn probe_memory() -> Result<MemoryCapabilities> {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_all();

        let total_bytes = sys.total_memory();
        let available_bytes = sys.available_memory();
        
        // Estimate memory bandwidth (simplified)
        // DDR4-3200 ≈ 25,600 MB/s, DDR5-4800 ≈ 38,400 MB/s
        let bandwidth_mbps = 25600;

        Ok(MemoryCapabilities {
            total_bytes,
            available_bytes,
            bandwidth_mbps,
        })
    }

    /// Probe storage capabilities.
    pub fn probe_storage() -> Result<StorageCapabilities> {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_all();

        let disks = sys.disks();
        
        let total_bytes: u64 = disks.iter().map(|d| d.total_space()).sum();
        let available_bytes: u64 = disks.iter().map(|d| d.available_space()).sum();
        
        // Estimate NVMe throughput (conservative estimate)
        let read_throughput_mbps = 3500;
        let write_throughput_mbps = 3000;

        Ok(StorageCapabilities {
            total_bytes,
            available_bytes,
            read_throughput_mbps,
            write_throughput_mbps,
        })
    }

    /// Probe network capabilities.
    pub fn probe_network() -> Result<NetworkCapabilities> {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_all();

        let interfaces = sys.networks()
            .iter()
            .filter(|(_, iface)| !iface.name().contains("Loopback"))
            .map(|(name, iface)| NetworkInterface {
                name: name.clone(),
                ip_address: iface.ip().to_string(),
                mac_address: iface.mac().to_string(),
                bandwidth_mbps: 1000, // Default to 1 Gbps estimate
                supports_rdma: false,
            })
            .collect();

        Ok(NetworkCapabilities { interfaces })
    }

    /// Detect supported runtime backends.
    pub fn detect_supported_runtimes() -> Vec<RuntimeBackend> {
        let mut runtimes = vec![RuntimeBackend::CPU];

        // Always add GGML as it's CPU-based
        runtimes.push(RuntimeBackend::GGML);

        // Detect CUDA availability
        #[cfg(feature = "nvidia")]
        {
            if Self::has_cuda() {
                runtimes.push(RuntimeBackend::CUDA);
            }
        }

        // Detect ROCm on Linux
        #[cfg(target_os = "linux")]
        {
            if Self::has_rocm() {
                runtimes.push(RuntimeBackend::ROCm);
            }
        }

        // Detect MLX on Apple Silicon
        #[cfg(target_os = "macos")]
        {
            if Self::has_apple_silicon() {
                runtimes.push(RuntimeBackend::MLX);
            }
        }

        // Vulkan is generally available
        runtimes.push(RuntimeBackend::Vulkan);

        runtimes
    }

    #[cfg(feature = "nvidia")]
    fn has_cuda() -> bool {
        // Check for CUDA libraries
        std::path::Path::new("/usr/local/cuda").exists() 
            || std::env::var("CUDA_PATH").is_ok()
    }

    #[cfg(target_os = "linux")]
    fn has_rocm() -> bool {
        std::path::Path::new("/opt/rocm").exists()
    }

    #[cfg(target_os = "macos")]
    fn has_apple_silicon() -> bool {
        std::env::consts::ARCH == "aarch64"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_cpu() {
        let cpu = HardwareProbe::probe_cpu().unwrap();
        assert!(cpu.cores > 0);
        assert!(cpu.threads > 0);
        assert!(!cpu.architecture.is_empty());
    }

    #[test]
    fn test_probe_memory() {
        let memory = HardwareProbe::probe_memory().unwrap();
        assert!(memory.total_bytes > 0);
        assert!(memory.available_bytes > 0);
        assert!(memory.available_bytes <= memory.total_bytes);
    }

    #[test]
    fn test_probe_all() {
        let capabilities = HardwareProbe::probe_all().unwrap();
        assert!(!capabilities.supported_runtimes.is_empty());
    }
}