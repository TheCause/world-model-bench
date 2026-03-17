#!/bin/bash
# =============================================================================
# Micro HTTP server for remote monitoring
# =============================================================================
# Serves /workspace/results/ on port 8080
# Access: https://<pod-id>-8080.proxy.runpod.net/
# =============================================================================

RESULTS_DIR="/workspace/results"
PORT=8080

mkdir -p "$RESULTS_DIR"

# Generate a simple index page
cat > "$RESULTS_DIR/index.html" <<'INDEXEOF'
<html><head><title>world-model-bench — Status</title>
<meta http-equiv="refresh" content="30">
<style>body{font-family:monospace;margin:2em;background:#1a1a2e;color:#e0e0e0}
a{color:#4fc3f7}pre{background:#16213e;padding:1em;overflow-x:auto}</style>
</head><body>
<h1>world-model-bench — Status</h1>
<p>Auto-refresh every 30s. Files in /workspace/results/:</p>
<pre id="listing"></pre>
<script>
fetch('.').then(r=>r.text()).then(t=>{
  document.getElementById('listing').textContent=t;
});
</script>
</body></html>
INDEXEOF

echo "Status server starting on port $PORT (serving $RESULTS_DIR)"
cd "$RESULTS_DIR" && python3 -m http.server "$PORT" --bind 0.0.0.0
