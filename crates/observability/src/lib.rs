//! Observability utilities for logging, metrics, and distributed tracing.

use tracing::{info, warn, error, debug, instrument};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};
use opentelemetry::trace::TraceError;
use opentelemetry::sdk::trace as sdktrace;
use opentelemetry::sdk::resource::Resource;
use opentelemetry::KeyValue;

/// Initialize the observability stack.
pub fn init(service_name: &str) -> Result<(), TraceError> {
    // Set up tracing subscriber
    let env_filter = EnvFilter::from_default_env()
        .add_directive(tracing::Level::INFO.into())
        .add_directive("hyper=warn".into())
        .add_directive("tonic=warn".into());

    // Console logging layer
    let fmt_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(true)
        .with_level(true);

    // OpenTelemetry tracing layer (optional - requires Jaeger)
    let tracer = opentelemetry_jaeger::new_agent_pipeline()
        .with_service_name(service_name)
        .with_auto_split_batch(true)
        .init();

    let telemetry_layer = tracing_opentelemetry::layer().with_tracer(tracer);

    // Combine layers
    tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer)
        .with(telemetry_layer)
        .init();

    info!("Observability initialized for service: {}", service_name);

    Ok(())
}

/// Initialize observability without distributed tracing (for development).
pub fn init_simple(service_name: &str) {
    let env_filter = EnvFilter::from_default_env()
        .add_directive(tracing::Level::INFO.into())
        .add_directive("hyper=warn".into())
        .add_directive("tonic=warn".into());

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_thread_ids(true)
        .with_level(true)
        .init();

    info!("Observability initialized (simple mode) for service: {}", service_name);
}

/// Metrics collector for cluster operations.
#[derive(Debug, Clone)]
pub struct MetricsCollector {
    service_name: String,
}

impl MetricsCollector {
    pub fn new(service_name: String) -> Self {
        Self { service_name }
    }

    /// Record a counter metric.
    #[instrument(skip(self))]
    pub fn increment_counter(&self, name: &str, value: u64, labels: &[(&str, &str)]) {
        debug!(
            service = %self.service_name,
            metric = name,
            value = value,
            labels = ?labels,
            "counter incremented"
        );
        // In a real implementation, this would send to Prometheus/OTLP
    }

    /// Record a gauge metric.
    #[instrument(skip(self))]
    pub fn record_gauge(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        debug!(
            service = %self.service_name,
            metric = name,
            value = value,
            labels = ?labels,
            "gauge recorded"
        );
    }

    /// Record a histogram metric.
    #[instrument(skip(self))]
    pub fn record_histogram(&self, name: &str, value: f64, labels: &[(&str, &str)]) {
        debug!(
            service = %self.service_name,
            metric = name,
            value = value,
            labels = ?labels,
            "histogram recorded"
        );
    }

    /// Record timing for an operation.
    #[instrument(skip(self))]
    pub fn record_timing<F, R>(&self, name: &str, labels: &[(&str, &str)], f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let start = std::time::Instant::now();
        let result = f();
        let duration = start.elapsed().as_secs_f64();
        
        self.record_histogram(format!("{}_duration", name).as_str(), duration, labels);
        result
    }
}

/// Structured logging context for operations.
#[derive(Debug)]
pub struct LogContext {
    operation: String,
    node_id: Option<String>,
    session_id: Option<String>,
    model_id: Option<String>,
}

impl LogContext {
    pub fn new(operation: String) -> Self {
        Self {
            operation,
            node_id: None,
            session_id: None,
            model_id: None,
        }
    }

    pub fn with_node_id(mut self, node_id: String) -> Self {
        self.node_id = Some(node_id);
        self
    }

    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = Some(session_id);
        self
    }

    pub fn with_model_id(mut self, model_id: String) -> Self {
        self.model_id = Some(model_id);
        self
    }

    pub fn info(&self, message: &str) {
        info!(
            operation = %self.operation,
            node_id = self.node_id,
            session_id = self.session_id,
            model_id = self.model_id,
            "{}", message
        );
    }

    pub fn warn(&self, message: &str) {
        warn!(
            operation = %self.operation,
            node_id = self.node_id,
            session_id = self.session_id,
            model_id = self.model_id,
            "{}", message
        );
    }

    pub fn error(&self, message: &str) {
        error!(
            operation = %self.operation,
            node_id = self.node_id,
            session_id = self.session_id,
            model_id = self.model_id,
            "{}", message
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_context() {
        let ctx = LogContext::new("test_operation".to_string())
            .with_node_id("node-123".to_string())
            .with_session_id("session-456".to_string());
        
        ctx.info("Test message");
    }

    #[test]
    fn test_metrics_collector() {
        let metrics = MetricsCollector::new("test-service".to_string());
        metrics.increment_counter("test_counter", 1, &[]);
        metrics.record_gauge("test_gauge", 42.0, &[]);
        
        let result = metrics.record_timing("test_operation", &[], || {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
    }
}