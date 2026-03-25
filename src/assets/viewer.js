// Ferroviz interactive graph viewer
// Embedded in self-contained HTML output

(function() {
    "use strict";

    const data = window.__FERROVIZ_DATA__;
    if (!data) {
        document.getElementById("graph-container").innerHTML = "<p>No graph data found.</p>";
        return;
    }

    const container = document.getElementById("graph-container");
    const detailPanel = document.getElementById("detail-panel");
    const timelineContainer = document.getElementById("timeline-container");

    // Color scheme
    const COLORS = {
        IO: "#9e9e9e",
        Elementwise: "#42a5f5",
        Reduction: "#ab47bc",
        MatMul: "#26a69a",
        Linear: "#66bb6a",
        Activation: "#ffa726",
        Shape: "#bdbdbd",
        Control: "#ef5350",
        Fused: "#7e57c2",
        // Special states
        cpuRoundTrip: "#e53935",
        deviceTransition: "#ff9800",
        fusedBorder: "#7e57c2",
    };

    // Build the graph using dagre-d3's bundled graphlib
    const g = new dagreD3.graphlib.Graph({ compound: true });
    g.setGraph({ rankdir: "TB", ranksep: 60, nodesep: 30, marginx: 20, marginy: 20 });
    g.setDefaultEdgeLabel(function() { return {}; });

    // Create cluster groups for fusion clusters
    data.fusion_groups.forEach(function(cluster) {
        g.setNode("cluster_" + cluster.id, {
            label: cluster.kind,
            clusterLabelPos: "top",
            style: "fill: " + COLORS.fusedBorder + "22; stroke: " + COLORS.fusedBorder + ";",
        });
    });

    // Add nodes
    data.nodes.forEach(function(node) {
        const hasCpuRoundTrip = node.cuda_memcpy && node.cuda_memcpy.length > 0 &&
            node.cuda_kernels && node.cuda_kernels.length === 0;
        const hasDeviceTransition = node.observed_input_devices && node.observed_output_device &&
            node.observed_input_devices.some(function(d) {
                return JSON.stringify(d) !== JSON.stringify(node.observed_output_device);
            });

        let color = COLORS[node.category] || COLORS.IO;
        if (hasCpuRoundTrip) color = COLORS.cpuRoundTrip;
        else if (hasDeviceTransition) color = COLORS.deviceTransition;

        const shapeStr = node.output_shapes.map(function(s) { return "[" + s.join("×") + "]"; }).join(", ");
        const timeStr = node.observed_duration_us != null ? " " + node.observed_duration_us + "µs" : "";
        const label = node.op_label + (shapeStr ? "\n" + shapeStr : "") + timeStr;

        g.setNode("n" + node.id, {
            label: label,
            style: "fill: " + color + "; stroke: #333; cursor: pointer;",
            labelStyle: "fill: #fff; font-size: 12px;",
            shape: "rect",
            rx: 5,
            ry: 5,
            padding: 8,
            id: "node-" + node.id,
            _data: node,
        });

        if (node.cluster_id != null) {
            g.setParent("n" + node.id, "cluster_" + node.cluster_id);
        }
    });

    // Add edges
    data.edges.forEach(function(edge, i) {
        const shapeStr = edge.shape.length > 0 ? edge.shape.join("×") : "";
        const sizeStr = edge.size_bytes > 0 ? formatBytes(edge.size_bytes) : "";
        const label = [shapeStr, sizeStr].filter(Boolean).join(" ");
        let style = "stroke: #666; stroke-width: 1.5px;";
        if (edge.observed_device_transition === "CpuToGpu") {
            style = "stroke: " + COLORS.deviceTransition + "; stroke-width: 2.5px;";
        } else if (edge.observed_device_transition === "GpuToCpu") {
            style = "stroke: " + COLORS.cpuRoundTrip + "; stroke-width: 2.5px;";
        }

        g.setEdge("n" + edge.from_node, "n" + edge.to_node, {
            label: label,
            labelStyle: "fill: #999; font-size: 10px;",
            style: style,
            arrowheadStyle: "fill: #666;",
            curve: d3.curveBasis,
        });
    });

    // Render with dagre-d3
    const svg = d3.select("#graph-svg");
    const inner = svg.append("g");
    const render = new dagreD3.render();
    render(inner, g);

    // Fit to container
    const graphWidth = g.graph().width || 800;
    const graphHeight = g.graph().height || 600;
    svg.attr("viewBox", "0 0 " + (graphWidth + 40) + " " + (graphHeight + 40));

    // Zoom/pan (d3 v5 uses d3.event, not a callback parameter)
    const zoom = d3.zoom().scaleExtent([0.1, 4]).on("zoom", function() {
        inner.attr("transform", d3.event.transform);
    });
    svg.call(zoom);

    // Initial fit
    const containerRect = container.getBoundingClientRect();
    const scale = Math.min(
        containerRect.width / (graphWidth + 40),
        containerRect.height / (graphHeight + 40),
        1
    );
    const tx = (containerRect.width - graphWidth * scale) / 2;
    const ty = 20;
    svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));

    // Node click handler
    svg.selectAll("g.node").on("click", function(event) {
        const nodeId = this.id.replace("node-", "");
        const node = data.nodes.find(function(n) { return n.id === parseInt(nodeId); });
        if (node) showDetail(node);
    });

    function showDetail(node) {
        let html = "<h3>" + escapeHtml(node.op_label) + "</h3>";
        html += "<table>";
        html += row("Category", node.category);
        html += row("Node ID", node.id);
        if (node.output_shapes.length > 0) {
            html += row("Output shapes", node.output_shapes.map(function(s) { return "[" + s.join(", ") + "]"; }).join("; "));
        }
        if (node.observed_output_device != null) {
            html += row("Device", formatDevice(node.observed_output_device));
        }
        if (node.observed_duration_us != null) {
            html += row("Duration", node.observed_duration_us + " µs");
        }
        if (node.requires_grad != null) {
            html += row("Requires grad", node.requires_grad ? "yes" : "no");
        }
        if (node.cluster_id != null) {
            const cluster = data.fusion_groups.find(function(c) { return c.id === node.cluster_id; });
            html += row("Fusion group", cluster ? cluster.kind + " #" + cluster.id : "#" + node.cluster_id);
        }
        if (node.cuda_kernels && node.cuda_kernels.length > 0) {
            html += "<tr><td colspan=\"2\"><strong>CUDA Kernels</strong></td></tr>";
            node.cuda_kernels.forEach(function(k) {
                html += row("  " + k.kernel_name, (k.duration_ns / 1000).toFixed(1) + " µs, grid=[" + k.grid.join(",") + "], block=[" + k.block.join(",") + "]");
            });
        }
        if (node.cuda_memcpy && node.cuda_memcpy.length > 0) {
            html += "<tr><td colspan=\"2\"><strong>CUDA Memcpy</strong></td></tr>";
            node.cuda_memcpy.forEach(function(m) {
                html += row("  " + m.direction, formatBytes(m.bytes) + " in " + (m.duration_ns / 1000).toFixed(1) + " µs");
            });
        }
        html += "</table>";
        detailPanel.innerHTML = html;
        detailPanel.style.display = "block";
    }

    // Timeline view
    if (data.runtime && data.runtime.op_events.length > 0) {
        renderTimeline(data.runtime.op_events);
    }

    function renderTimeline(events) {
        const maxDuration = Math.max.apply(null, events.map(function(e) { return e.duration_us; }));
        if (maxDuration === 0) return;

        let html = "<h3>Op Timeline (µs)</h3><div class=\"timeline\">";
        events.forEach(function(e) {
            if (e.duration_us === 0 && e.op_label.startsWith("Input")) return;
            const width = Math.max(2, (e.duration_us / maxDuration) * 100);
            const color = COLORS[data.nodes.find(function(n) { return n.id === e.node_id; })?.category] || "#666";
            html += "<div class=\"timeline-bar\" style=\"width: " + width + "%; background: " + color + ";\" title=\"" +
                escapeHtml(e.op_label) + ": " + e.duration_us + "µs\">" +
                "<span class=\"timeline-label\">" + escapeHtml(e.op_label) + " " + e.duration_us + "µs</span></div>";
        });
        html += "</div>";
        timelineContainer.innerHTML = html;
    }

    // Helpers
    function row(key, value) {
        return "<tr><td>" + escapeHtml(String(key)) + "</td><td>" + escapeHtml(String(value)) + "</td></tr>";
    }

    function formatDevice(d) {
        if (typeof d === "string") return d;
        if (d === "Cpu") return "CPU";
        if (d && d.Cuda !== undefined) return "CUDA:" + d.Cuda;
        return JSON.stringify(d);
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function escapeHtml(s) {
        const div = document.createElement("div");
        div.textContent = s;
        return div.innerHTML;
    }
})();
