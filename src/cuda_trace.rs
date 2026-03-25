//! Parse nsys SQLite output and correlate CUDA events to graph ops.
//!
//! This module is only available when the `cuda-trace` feature is enabled.

#[cfg(feature = "cuda-trace")]
mod inner {
    use std::path::Path;

    use rusqlite::Connection;

    use crate::model::*;

    /// Correlate nsys profiling data with an existing VisGraph.
    ///
    /// The VisGraph must have runtime data (from `capture()`) so that per-op
    /// time windows are available for timestamp-based correlation.
    ///
    /// # Arguments
    /// * `vis` - A mutable VisGraph with runtime profiling data
    /// * `nsys_path` - Path to the nsys SQLite database (.sqlite or .nsys-rep)
    pub fn correlate_nsys(
        vis: &mut VisGraph,
        nsys_path: &Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let runtime = vis
            .runtime
            .as_ref()
            .ok_or("VisGraph must have runtime data for CUDA trace correlation")?;

        let conn = Connection::open(nsys_path)?;

        // Read kernel launches
        let kernels = read_kernel_launches(&conn)?;

        // Read memory copies
        let memcpys = read_memcpy_events(&conn)?;

        // Build cumulative time windows for each op based on instrumented timing
        let time_windows = build_time_windows(&runtime.op_events);

        // Correlate events to ops
        let mut total_kernel_time_ns: u64 = 0;
        let mut total_memcpy_time_ns: u64 = 0;
        let mut kernel_count: usize = 0;
        let mut memcpy_count: usize = 0;

        for node in &mut vis.nodes {
            if let Some(window) = time_windows.get(&node.id) {
                // Find kernels that launched during this op's time window
                for k in &kernels {
                    if k.start_ns >= window.start_ns && k.start_ns < window.end_ns {
                        total_kernel_time_ns += k.duration_ns;
                        kernel_count += 1;
                        node.cuda_kernels.push(CudaKernelLaunch {
                            kernel_name: k.name.clone(),
                            duration_ns: k.duration_ns,
                            grid: k.grid,
                            block: k.block,
                        });
                    }
                }

                // Find memcpy events during this op's time window
                for m in &memcpys {
                    if m.start_ns >= window.start_ns && m.start_ns < window.end_ns {
                        total_memcpy_time_ns += m.duration_ns;
                        memcpy_count += 1;
                        node.cuda_memcpy.push(CudaMemcpy {
                            direction: m.direction,
                            bytes: m.bytes,
                            duration_ns: m.duration_ns,
                        });
                    }
                }
            }
        }

        vis.cuda_trace = Some(CudaTrace {
            total_kernel_time_ns,
            total_memcpy_time_ns,
            kernel_count,
            memcpy_count,
        });

        Ok(())
    }

    struct RawKernel {
        name: String,
        start_ns: u64,
        duration_ns: u64,
        grid: [u32; 3],
        block: [u32; 3],
    }

    struct RawMemcpy {
        direction: MemcpyDirection,
        start_ns: u64,
        duration_ns: u64,
        bytes: usize,
    }

    struct TimeWindow {
        start_ns: u64,
        end_ns: u64,
    }

    fn read_kernel_launches(
        conn: &Connection,
    ) -> Result<Vec<RawKernel>, Box<dyn std::error::Error>> {
        let mut stmt = conn.prepare(
            "SELECT
                demangledName,
                start,
                end,
                gridX, gridY, gridZ,
                blockX, blockY, blockZ
             FROM CUPTI_ACTIVITY_KIND_KERNEL
             ORDER BY start",
        )?;

        let rows = stmt.query_map([], |row| {
            let name: String = row.get(0)?;
            let start: i64 = row.get(1)?;
            let end: i64 = row.get(2)?;
            let grid_x: u32 = row.get(3)?;
            let grid_y: u32 = row.get(4)?;
            let grid_z: u32 = row.get(5)?;
            let block_x: u32 = row.get(6)?;
            let block_y: u32 = row.get(7)?;
            let block_z: u32 = row.get(8)?;

            Ok(RawKernel {
                name,
                start_ns: start as u64,
                duration_ns: (end - start) as u64,
                grid: [grid_x, grid_y, grid_z],
                block: [block_x, block_y, block_z],
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    fn read_memcpy_events(
        conn: &Connection,
    ) -> Result<Vec<RawMemcpy>, Box<dyn std::error::Error>> {
        let mut stmt = conn.prepare(
            "SELECT
                copyKind,
                start,
                end,
                bytes
             FROM CUPTI_ACTIVITY_KIND_MEMCPY
             ORDER BY start",
        )?;

        let rows = stmt.query_map([], |row| {
            let kind: i32 = row.get(0)?;
            let start: i64 = row.get(1)?;
            let end: i64 = row.get(2)?;
            let bytes: i64 = row.get(3)?;

            let direction = match kind {
                1 => MemcpyDirection::HostToDevice,
                2 => MemcpyDirection::DeviceToHost,
                8 => MemcpyDirection::DeviceToDevice,
                _ => MemcpyDirection::DeviceToHost, // fallback
            };

            Ok(RawMemcpy {
                direction,
                start_ns: start as u64,
                duration_ns: (end - start) as u64,
                bytes: bytes as usize,
            })
        })?;

        Ok(rows.filter_map(|r| r.ok()).collect())
    }

    /// Build time windows from op events by accumulating durations.
    /// Assumes ops run sequentially in the order recorded.
    fn build_time_windows(
        events: &[OpEvent],
    ) -> std::collections::HashMap<usize, TimeWindow> {
        let mut map = std::collections::HashMap::new();
        let mut cursor_ns: u64 = 0;

        for event in events {
            let duration_ns = event.duration_us * 1000;
            map.insert(
                event.node_id,
                TimeWindow {
                    start_ns: cursor_ns,
                    end_ns: cursor_ns + duration_ns,
                },
            );
            cursor_ns += duration_ns;
        }

        map
    }
}

#[cfg(feature = "cuda-trace")]
pub use inner::correlate_nsys;
