// Ferroviz interactive graph viewer
// Embedded in self-contained HTML output

(function() {
    "use strict";

    var forwardData = window.__FERROVIZ_DATA__;
    var backwardData = window.__FERROVIZ_BACKWARD__;

    if (!forwardData) {
        document.getElementById("graph-container").innerHTML = "<p>No graph data found.</p>";
        return;
    }

    // Show tab bar if we have backward data
    var hasBackward = backwardData && backwardData !== null && backwardData.nodes && backwardData.nodes.length > 0;
    if (hasBackward) {
        document.getElementById("tab-bar").style.display = "flex";
    }

    var currentTab = "forward";
    var container = document.getElementById("graph-container");
    var detailPanel = document.getElementById("detail-panel");
    var timelineContainer = document.getElementById("timeline-container");

    // Category base colors
    var CATEGORY_COLORS = {
        IO: "#9e9e9e",
        Elementwise: "#42a5f5",
        Reduction: "#ab47bc",
        MatMul: "#26a69a",
        Linear: "#66bb6a",
        Activation: "#ffa726",
        Shape: "#bdbdbd",
        Control: "#ef5350",
        Fused: "#7e57c2",
        Backward: "#f06292",
    };

    // Special state colors
    var SPECIAL = {
        cpuRoundTrip: "#e53935",
        gpuFallback: "#ff9800",
        fusedBorder: "#7e57c2",
    };

    // Tab switching
    document.querySelectorAll(".tab").forEach(function(btn) {
        btn.addEventListener("click", function() {
            document.querySelectorAll(".tab").forEach(function(b) { b.classList.remove("active"); });
            btn.classList.add("active");
            currentTab = btn.getAttribute("data-tab");
            renderGraph(currentTab === "backward" ? backwardData : forwardData);
        });
    });

    // Initial render
    renderGraph(forwardData);

    function renderGraph(data) {
        // Clear previous
        container.innerHTML = '<svg id="graph-svg"></svg>';
        detailPanel.innerHTML = "";
        detailPanel.style.display = "none";
        timelineContainer.innerHTML = "";

        if (!data || !data.nodes || data.nodes.length === 0) {
            container.innerHTML = "<p style='color:#999;padding:20px;'>No nodes in this graph.</p>";
            return;
        }

        var g = new dagreD3.graphlib.Graph({ compound: true });
        g.setGraph({ rankdir: "TB", ranksep: 60, nodesep: 30, marginx: 20, marginy: 20 });
        g.setDefaultEdgeLabel(function() { return {}; });

        // Create cluster groups for fusion clusters
        if (data.fusion_groups) {
            data.fusion_groups.forEach(function(cluster) {
                g.setNode("cluster_" + cluster.id, {
                    label: cluster.kind,
                    clusterLabelPos: "top",
                    style: "fill: " + SPECIAL.fusedBorder + "22; stroke: " + SPECIAL.fusedBorder + ";",
                });
            });
        }

        // Add nodes with device-aware styling
        data.nodes.forEach(function(node) {
            var style = getNodeStyle(node);
            var shapeStr = node.output_shapes.map(function(s) { return "[" + s.join("\u00d7") + "]"; }).join(", ");
            var timeStr = node.observed_duration_us != null ? " " + node.observed_duration_us + "\u00b5s" : "";
            var label = node.op_label + (shapeStr ? "\n" + shapeStr : "") + timeStr;

            g.setNode("n" + node.id, {
                label: label,
                style: style.nodeStyle,
                labelStyle: style.labelStyle,
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
        data.edges.forEach(function(edge) {
            var shapeStr = edge.shape.length > 0 ? edge.shape.join("\u00d7") : "";
            var sizeStr = edge.size_bytes > 0 ? formatBytes(edge.size_bytes) : "";
            var label = [shapeStr, sizeStr].filter(Boolean).join(" ");
            var style = "stroke: #555; stroke-width: 1.5px;";
            var arrowStyle = "fill: #555;";

            if (edge.observed_device_transition === "CpuToGpu") {
                style = "stroke: #66bb6a; stroke-width: 3px;";
                arrowStyle = "fill: #66bb6a;";
                label = "\u2191 GPU  " + label;
            } else if (edge.observed_device_transition === "GpuToCpu") {
                style = "stroke: " + SPECIAL.cpuRoundTrip + "; stroke-width: 3px;";
                arrowStyle = "fill: " + SPECIAL.cpuRoundTrip + ";";
                label = "\u2193 CPU  " + label;
            }

            g.setEdge("n" + edge.from_node, "n" + edge.to_node, {
                label: label,
                labelStyle: "fill: #999; font-size: 10px;",
                style: style,
                arrowheadStyle: arrowStyle,
                curve: d3.curveBasis,
            });
        });

        // Render with dagre-d3
        var svg = d3.select("#graph-svg");
        var inner = svg.append("g");
        var render = new dagreD3.render();
        render(inner, g);

        // Fit to container
        var graphWidth = g.graph().width || 800;
        var graphHeight = g.graph().height || 600;
        svg.attr("viewBox", "0 0 " + (graphWidth + 40) + " " + (graphHeight + 40));

        // Zoom/pan
        var zoom = d3.zoom().scaleExtent([0.1, 4]).on("zoom", function() {
            inner.attr("transform", d3.event.transform);
        });
        svg.call(zoom);

        // Initial fit
        var containerRect = container.getBoundingClientRect();
        var scale = Math.min(
            containerRect.width / (graphWidth + 40),
            containerRect.height / (graphHeight + 40),
            1
        );
        var tx = (containerRect.width - graphWidth * scale) / 2;
        var ty = 20;
        svg.call(zoom.transform, d3.zoomIdentity.translate(tx, ty).scale(scale));

        // Node click handler
        svg.selectAll("g.node").on("click", function() {
            var nodeId = this.id.replace("node-", "");
            var node = data.nodes.find(function(n) { return n.id === parseInt(nodeId); });
            if (node) showDetail(node, data);
        });

        // Timeline view
        if (data.runtime && data.runtime.op_events.length > 0) {
            renderTimeline(data.runtime.op_events, data);
        }
    }

    /**
     * Determine node styling based on device state.
     *
     * GPU nodes: solid filled background with category color (saturated, opaque)
     * CPU nodes: outline only — transparent fill with colored border
     * GPU fallback: solid fill with warning color + dashed border
     * No runtime data: muted solid fill (we don't know the device)
     */
    function getNodeStyle(node) {
        var baseColor = CATEGORY_COLORS[node.category] || CATEGORY_COLORS.IO;
        var isOnGpu = false;
        var isOnCpu = false;
        var isFallback = node.gpu_fallback === true;

        // Check for CPU round-trip (has memcpy but no kernels — from nsys data)
        var hasCpuRoundTrip = node.cuda_memcpy && node.cuda_memcpy.length > 0 &&
            node.cuda_kernels && node.cuda_kernels.length === 0;

        if (node.observed_output_device != null) {
            if (node.observed_output_device === "Cpu") {
                isOnCpu = true;
            } else if (typeof node.observed_output_device === "object" &&
                       node.observed_output_device.Cuda !== undefined) {
                isOnGpu = true;
            }
        }

        var nodeStyle, labelStyle;

        if (hasCpuRoundTrip) {
            // CPU round-trip detected via nsys: filled red
            nodeStyle = "fill: " + SPECIAL.cpuRoundTrip + "; stroke: #b71c1c; stroke-width: 2px; cursor: pointer;";
            labelStyle = "fill: #fff; font-size: 12px;";
        } else if (isFallback) {
            // GPU fallback: filled warning color with dashed border
            nodeStyle = "fill: " + SPECIAL.gpuFallback + "; stroke: #e65100; stroke-width: 2px; stroke-dasharray: 4,2; cursor: pointer;";
            labelStyle = "fill: #fff; font-size: 12px;";
        } else if (isOnGpu) {
            // GPU: solid filled with category color
            nodeStyle = "fill: " + baseColor + "; stroke: " + baseColor + "; stroke-width: 2px; cursor: pointer;";
            labelStyle = "fill: #fff; font-size: 12px;";
        } else if (isOnCpu) {
            // CPU: outline only — transparent fill, colored border
            nodeStyle = "fill: #1a1a2e; stroke: " + baseColor + "; stroke-width: 2px; cursor: pointer;";
            labelStyle = "fill: " + baseColor + "; font-size: 12px;";
        } else {
            // No runtime data: muted fill
            nodeStyle = "fill: " + baseColor + "66; stroke: " + baseColor + "; stroke-width: 1px; cursor: pointer;";
            labelStyle = "fill: #ccc; font-size: 12px;";
        }

        return { nodeStyle: nodeStyle, labelStyle: labelStyle };
    }

    function showDetail(node, data) {
        var html = "<h3>" + escapeHtml(node.op_label) + "</h3>";
        html += "<table>";
        html += row("Category", node.category);
        html += row("Node ID", node.id);
        if (node.output_shapes.length > 0) {
            html += row("Output shapes", node.output_shapes.map(function(s) { return "[" + s.join(", ") + "]"; }).join("; "));
        }
        if (node.observed_output_device != null) {
            var devStr = formatDevice(node.observed_output_device);
            if (node.gpu_fallback) {
                devStr += " (GPU FALLBACK \u2014 missing kernel)";
            }
            html += row("Device", devStr);
        }
        if (node.observed_input_devices && node.observed_input_devices.length > 0) {
            html += row("Input devices", node.observed_input_devices.map(formatDevice).join(", "));
        }
        if (node.observed_duration_us != null) {
            html += row("Duration", node.observed_duration_us + " \u00b5s");
        }
        if (node.requires_grad != null) {
            html += row("Requires grad", node.requires_grad ? "yes" : "no");
        }
        if (node.gpu_fallback) {
            html += row("GPU fallback", "Yes \u2014 op ran on CPU due to missing GPU kernel");
        }
        if (node.cluster_id != null && data.fusion_groups) {
            var cluster = data.fusion_groups.find(function(c) { return c.id === node.cluster_id; });
            html += row("Fusion group", cluster ? cluster.kind + " #" + cluster.id : "#" + node.cluster_id);
        }
        if (node.cuda_kernels && node.cuda_kernels.length > 0) {
            html += "<tr><td colspan=\"2\"><strong>CUDA Kernels</strong></td></tr>";
            node.cuda_kernels.forEach(function(k) {
                html += row("  " + k.kernel_name, (k.duration_ns / 1000).toFixed(1) + " \u00b5s, grid=[" + k.grid.join(",") + "], block=[" + k.block.join(",") + "]");
            });
        }
        if (node.cuda_memcpy && node.cuda_memcpy.length > 0) {
            html += "<tr><td colspan=\"2\"><strong>CUDA Memcpy</strong></td></tr>";
            node.cuda_memcpy.forEach(function(m) {
                html += row("  " + m.direction, formatBytes(m.bytes) + " in " + (m.duration_ns / 1000).toFixed(1) + " \u00b5s");
            });
        }
        html += "</table>";
        detailPanel.innerHTML = html;
        detailPanel.style.display = "block";
    }

    function renderTimeline(events, data) {
        var maxDuration = Math.max.apply(null, events.map(function(e) { return e.duration_us; }));
        if (maxDuration === 0) return;

        var html = "<h3>Op Timeline (\u00b5s)</h3><div class=\"timeline\">";
        events.forEach(function(e) {
            if (e.duration_us === 0 && e.op_label.startsWith("Input")) return;
            var width = Math.max(2, (e.duration_us / maxDuration) * 100);
            var node = data.nodes.find(function(n) { return n.id === e.node_id; });
            var color = CATEGORY_COLORS[node ? node.category : "IO"] || "#666";
            if (e.gpu_fallback) color = SPECIAL.gpuFallback;

            var devTag = "";
            if (e.output_device === "Cpu") {
                devTag = " [CPU]";
            } else if (typeof e.output_device === "object" && e.output_device.Cuda !== undefined) {
                devTag = " [GPU]";
            }
            if (e.gpu_fallback) devTag = " [FALLBACK]";

            html += "<div class=\"timeline-bar\" style=\"width: " + width + "%; background: " + color + ";\" title=\"" +
                escapeHtml(e.op_label) + ": " + e.duration_us + "\u00b5s" + devTag + "\">" +
                "<span class=\"timeline-label\">" + escapeHtml(e.op_label) + " " + e.duration_us + "\u00b5s" + devTag + "</span></div>";
        });
        html += "</div>";
        timelineContainer.innerHTML = html;
    }

    // Helpers
    function row(key, value) {
        return "<tr><td>" + escapeHtml(String(key)) + "</td><td>" + escapeHtml(String(value)) + "</td></tr>";
    }

    function formatDevice(d) {
        if (typeof d === "string") {
            if (d === "Cpu") return "CPU";
            return d;
        }
        if (d && d.Cuda !== undefined) return "CUDA:" + d.Cuda;
        return JSON.stringify(d);
    }

    function formatBytes(bytes) {
        if (bytes < 1024) return bytes + " B";
        if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function escapeHtml(s) {
        var div = document.createElement("div");
        div.textContent = s;
        return div.innerHTML;
    }
})();
