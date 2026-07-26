# TensorGrid - Distributed Unified AI Compute Runtime

A self-hosted AI runtime that combines the resources of multiple computers connected to the same local network. The system pools CPU, RAM, GPU, and VRAM across authorized nodes to run large AI models that cannot fit on a single machine.

## Architecture Overview

The system consists of:

- **Coordinator**: Central control plane service that manages cluster state, node registration, and model planning
- **Node Agent**: Lightweight service running on each participating computer for hardware detection and resource management
- **Runtime Adapters**: Pluggable backends for different inference engines (GGML, CUDA, ROCm, MLX, etc.)
- **Topology Profiler**: Network benchmarking and cluster topology measurement
- **Model Parser**: Support for GGUF and other model formats

## Project Structure

```
tensorgrid/
├── apps/              # Desktop, web, and CLI applications
├── services/          # Core services
│   ├── coordinator/   # Cluster coordinator service
│   ├── node-agent/    # Node agent service
│   ├── scheduler/     # Distributed scheduler
│   ├── planner/       # Execution planner
│   ├── tensor-directory/  # Tensor location tracking
│   └── model-registry/    # Model cache and registry
├── runtimes/          # Runtime backend adapters
│   ├── ggml/          # GGML/llama.cpp adapter
│   ├── cuda/          # CUDA adapter
│   ├── rocm/          # ROCm adapter
│   ├── mlx/           # Apple MLX adapter
│   ├── vulkan/        # Vulkan adapter
│   └── cpu/           # CPU-only adapter
├── transports/        # Data plane transports
│   ├── tcp/           # TCP transport
│   ├── quic/          # QUIC transport
│   ├── rdma/          # RDMA transport
│   ├── shared-memory/ # Shared memory transport
│   └── collectives/   # Collective operations (NCCL, RCCL)
├── protocols/         # gRPC protocol definitions
│   ├── control.proto  # Node control protocol
│   ├── execution.proto # Distributed execution protocol
│   ├── memory.proto   # Remote memory protocol
│   └── metrics.proto  # Metrics collection protocol
├── crates/            # Shared libraries
│   ├── cluster-types/ # Common data structures
│   ├── hardware-probe/ # Hardware detection
│   ├── secure-pairing/ # Node authentication
│   ├── model-format/   # Model format parsers
│   ├── topology/      # Network profiling
│   └── observability/ # Logging and metrics
├── tests/             # Integration and performance tests
└── docs/              # Documentation
```

## Technology Stack

- **Language**: Rust
- **Async Runtime**: Tokio
- **RPC**: gRPC with Tonic
- **Serialization**: Serde
- **Protocols**: Protocol Buffers
- **Security**: Rustls for TLS, ed25519-dalek for authentication
- **Observability**: Tracing, OpenTelemetry

## Building

### Prerequisites

- Rust 1.70 or later
- Protocol Buffers compiler (protoc)
- For GPU support: CUDA, ROCm, or appropriate drivers

### Build Steps

1. Clone the repository:
```bash
git clone https://github.com/tensorgrid/tensorgrid.git
cd tensorgrid
```

2. Build the workspace:
```bash
cargo build --release
```

3. Build specific services:
```bash
cargo build --release -p coordinator
cargo build --release -p node-agent
```

## Running

### Start the Coordinator

```bash
cargo run --release -p coordinator
```

The coordinator will start:
- gRPC server on `http://[::1]:50051` for node communication
- HTTP API on `http://[::1]:8080` for cluster management and OpenAI-compatible endpoints

### Start a Node Agent

```bash
COORDINATOR_ADDR=http://[::1]:50051 cargo run --release -p node-agent
```

## Development Status

This is an early-stage implementation. The following components are currently implemented:

### Core Infrastructure
- ✅ Repository structure and workspace setup
- ✅ Core data structures (cluster-types)
- ✅ Protocol buffer definitions (control, execution, memory, metrics)
- ✅ Protobuf build integration with tonic-build

### Shared Libraries (crates/)
- ✅ Hardware probing (CPU, memory, storage, network)
- ✅ Secure pairing with challenge-response authentication
- ✅ Topology profiler for network benchmarking
- ✅ GGUF model parser
- ✅ Observability (logging, metrics, distributed tracing)

### Services
- ✅ Coordinator service with gRPC server and HTTP API
- ✅ Node agent service with gRPC client and resource management
- ✅ Tensor directory service for tracking tensor locations
- ✅ Model registry service for model caching and distribution
- ✅ Scheduler service for resource allocation and admission control
- ✅ Planner service for execution plan generation
- ✅ Pipeline executor for distributed execution
- ✅ Fine-tuning service for model customization

### Runtimes
- ✅ GGML runtime adapter with tokenizer integration
- ✅ CPU runtime adapter stub
- ✅ CUDA runtime backend for NVIDIA GPUs
- ✅ ROCm runtime backend for AMD GPUs
- ✅ MLX runtime backend for Apple Silicon

### Transports
- ✅ TCP transport for tensor data transfer
- ✅ QUIC transport for improved performance
- ✅ RDMA transport for ultra-low-latency transfers

### Testing
- ✅ Integration tests for coordinator-node communication
- ✅ Unit tests for core components

### Resilience
- ✅ Heartbeat monitoring for node failure detection
- ✅ Automatic reconnection in node agent
- ✅ Graceful node drain and offline handling

### Desktop UI
- ✅ Tauri-based desktop application
- ✅ Cluster overview dashboard
- ✅ Node status monitoring
- ✅ Model listing

### Web UI
- ✅ Browser-based interface
- ✅ Cluster overview dashboard
- ✅ Node status monitoring
- ✅ Interactive chat interface

### Next Steps

- Integrate actual llama.cpp bindings for real inference
- Add comprehensive model download and management
- Implement advanced scheduling policies
- Add distributed training support
- Add model quantization and conversion tools
- Implement multi-GPU training support
- Add LoRA adapter management
- Create model evaluation and benchmarking tools

## Design Philosophy

The system is designed with these principles:

1. **Honest resource reporting**: Remote RAM is not presented as identical to local RAM. The system uses a distributed tensor runtime with unified resource namespace.

2. **Topology-aware planning**: Execution plans are based on measured network performance, not advertised specifications.

3. **Security-first**: All nodes require explicit pairing, mutual TLS authentication, and capability tokens.

4. **Heterogeneous support**: The system supports mixing different GPU vendors, CPU architectures, and runtime backends.

5. **Graceful degradation**: The system handles node failures, thermal throttling, and network issues with clear user communication.

## License

MIT OR Apache-2.0

## Contributing

This project is in early development. Contributions are welcome once the core architecture is stabilized.

## Documentation

See [ιδεα.md](ιδεα.md) for the complete technical design document.
