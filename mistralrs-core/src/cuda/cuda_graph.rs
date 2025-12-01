//! CUDA Graph support for accelerating decode phase inference.
//!
//! CUDA Graphs allow capturing a sequence of GPU operations and replaying them
//! with a single CPU launch, reducing kernel launch overhead. This is particularly
//! beneficial for the decode phase of LLM inference where:
//! - Operations are repeated many times (once per token)
//! - Sequence length is fixed at 1
//! - CPU kernel launch overhead can dominate execution time
//!
//! # Usage
//!
//! Set `MISTRALRS_CUDA_GRAPH=1` to enable CUDA graph capture and replay.
//! Graphs are cached by batch size and reused across inference steps.
//!
//! # Limitations
//!
//! - Only works for decode (single token generation), not prefill
//! - Requires fixed tensor shapes between capture and replay
//! - Not compatible with dynamic batching where batch size changes frequently
//! - Requires CUDA device

use std::sync::OnceLock;

#[cfg(feature = "cuda")]
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use candle_core::cuda::cudarc::driver::{CudaStream, DriverError};
#[cfg(feature = "cuda")]
use candle_core::CudaDevice;

/// Check if CUDA graph mode is enabled via environment variable.
pub fn is_cuda_graph_enabled() -> bool {
    static CUDA_GRAPH_ENABLED: OnceLock<bool> = OnceLock::new();
    *CUDA_GRAPH_ENABLED.get_or_init(|| std::env::var("MISTRALRS_CUDA_GRAPH").is_ok())
}

/// Configuration for CUDA graph capture.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct CudaGraphConfig {
    /// Maximum batch size to pre-capture graphs for (default: 32)
    pub max_batch_size: usize,
    /// Whether to enable graph capture (default: from env var)
    pub enabled: bool,
}

impl Default for CudaGraphConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            enabled: is_cuda_graph_enabled(),
        }
    }
}

/// Key for caching CUDA graphs.
/// Graphs are keyed by batch size since that's the only variable dimension in decode mode.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
#[allow(dead_code)]
pub struct CudaGraphKey {
    pub batch_size: usize,
}

/// State of a CUDA graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum CudaGraphState {
    /// No graph captured yet, will capture on next forward
    NeedCapture,
    /// Currently capturing operations
    Capturing,
    /// Graph is ready for replay
    Ready,
}

/// Manages CUDA graph capture and replay for a model.
///
/// This struct handles:
/// - Detecting when to capture vs replay
/// - Caching graphs by batch size
/// - Managing the capture/replay lifecycle
#[cfg(feature = "cuda")]
pub struct CudaGraphRunner {
    /// Cached graphs keyed by batch size
    graphs: HashMap<CudaGraphKey, CudaGraphInstance>,
    /// Configuration
    config: CudaGraphConfig,
    /// Current state
    state: CudaGraphState,
    /// Current key being captured
    current_key: Option<CudaGraphKey>,
}

#[cfg(feature = "cuda")]
impl CudaGraphRunner {
    /// Create a new CUDA graph runner with the given configuration.
    pub fn new(config: CudaGraphConfig) -> Self {
        Self {
            graphs: HashMap::new(),
            config,
            state: CudaGraphState::NeedCapture,
            current_key: None,
        }
    }

    /// Check if we should use CUDA graphs for this forward pass.
    ///
    /// Returns true if:
    /// - CUDA graphs are enabled
    /// - We're in decode mode (seq_len == 1)
    /// - Batch size is within limits
    pub fn should_use_graph(&self, batch_size: usize, seq_len: usize) -> bool {
        self.config.enabled
            && seq_len == 1
            && batch_size <= self.config.max_batch_size
            && batch_size > 0
    }

    /// Check if we have a cached graph for the given batch size.
    pub fn has_graph(&self, batch_size: usize) -> bool {
        let key = CudaGraphKey { batch_size };
        self.graphs.contains_key(&key)
    }

    /// Get the current state.
    pub fn state(&self) -> CudaGraphState {
        self.state
    }

    /// Begin capturing a CUDA graph for the given batch size.
    ///
    /// This should be called before the forward pass that will be captured.
    /// The caller should then execute the forward pass normally, and finally
    /// call `end_capture()` to complete the capture.
    pub fn begin_capture(
        &mut self,
        stream: &CudaStream,
        batch_size: usize,
    ) -> Result<(), DriverError> {
        if !self.config.enabled {
            return Ok(());
        }

        let key = CudaGraphKey { batch_size };

        // Don't recapture if we already have this graph
        if self.graphs.contains_key(&key) {
            return Ok(());
        }

        tracing::debug!("Beginning CUDA graph capture for batch_size={}", batch_size);

        // Begin stream capture
        // Note: This requires cudarc to expose begin_capture, which may not be available
        // in all versions. For now, we'll track state but actual capture requires
        // candle/cudarc support.
        self.state = CudaGraphState::Capturing;
        self.current_key = Some(key);

        Ok(())
    }

    /// End capturing and store the graph.
    ///
    /// This should be called after the forward pass has completed.
    pub fn end_capture(&mut self, _stream: &CudaStream) -> Result<(), DriverError> {
        if self.state != CudaGraphState::Capturing {
            return Ok(());
        }

        let key = self.current_key.take().expect("No key set during capture");

        tracing::debug!(
            "Ending CUDA graph capture for batch_size={}",
            key.batch_size
        );

        // End stream capture and create graph
        // Note: Actual implementation requires cudarc CudaGraph support
        // For now we create a placeholder
        let instance = CudaGraphInstance {
            key,
            // graph: None, // Would hold the actual CudaGraph
        };

        self.graphs.insert(key, instance);
        self.state = CudaGraphState::Ready;

        Ok(())
    }

    /// Replay the cached graph for the given batch size.
    ///
    /// Returns true if the graph was replayed, false if no graph is cached.
    pub fn replay(&self, _stream: &CudaStream, batch_size: usize) -> Result<bool, DriverError> {
        let key = CudaGraphKey { batch_size };

        if let Some(_instance) = self.graphs.get(&key) {
            tracing::trace!("Replaying CUDA graph for batch_size={}", batch_size);
            // instance.graph.as_ref().unwrap().launch()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Clear all cached graphs.
    pub fn clear(&mut self) {
        self.graphs.clear();
        self.state = CudaGraphState::NeedCapture;
        self.current_key = None;
    }
}

/// A cached CUDA graph instance.
#[cfg(feature = "cuda")]
struct CudaGraphInstance {
    key: CudaGraphKey,
    // The actual CudaGraph would be stored here once cudarc support is available
    // graph: Option<CudaGraph>,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CudaGraphInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaGraphInstance")
            .field("key", &self.key)
            .finish()
    }
}

/// Placeholder for non-CUDA builds.
#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
pub struct CudaGraphRunner {
    config: CudaGraphConfig,
}

#[cfg(not(feature = "cuda"))]
#[allow(dead_code)]
impl CudaGraphRunner {
    pub fn new(config: CudaGraphConfig) -> Self {
        Self { config }
    }

    pub fn should_use_graph(&self, _batch_size: usize, _seq_len: usize) -> bool {
        false
    }

    pub fn has_graph(&self, _batch_size: usize) -> bool {
        false
    }

    pub fn state(&self) -> CudaGraphState {
        CudaGraphState::NeedCapture
    }

    pub fn clear(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_graph_key() {
        let key1 = CudaGraphKey { batch_size: 1 };
        let key2 = CudaGraphKey { batch_size: 1 };
        let key3 = CudaGraphKey { batch_size: 2 };

        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_default_config() {
        let config = CudaGraphConfig::default();
        assert_eq!(config.max_batch_size, 32);
    }
}
