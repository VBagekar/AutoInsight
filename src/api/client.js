// Configurable via VITE_API_BASE_URL in a root .env file for production deployments.
const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

// ---------------------------------------------------------------------------
// Timeout helper — creates an AbortSignal that fires after `ms` milliseconds
// ---------------------------------------------------------------------------
const withTimeout = (ms) => {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), ms);
  return { signal: controller.signal, clear: () => clearTimeout(id) };
};

// ---------------------------------------------------------------------------
// Upload Dataset — 90s timeout, explicit error messages
// ---------------------------------------------------------------------------
export const uploadDataset = async (file) => {
  const { signal, clear } = withTimeout(120_000);
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
      signal,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Upload failed (HTTP ${response.status})`);
    }

    return await response.json();
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error('Upload timed out after 120 seconds. Try a smaller file or check your connection.');
    }
    throw err;
  } finally {
    clear();
  }
};

// ---------------------------------------------------------------------------
// Stream AI Query — 120s timeout, robust SSE parsing
// ---------------------------------------------------------------------------
export const streamAIQuery = async (query, datasetId, onThinking, onPayload, onError) => {
  const { signal, clear } = withTimeout(120_000);
  try {
    const response = await fetch(`${BASE_URL}/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query,
        dataset_id: datasetId,
      }),
      signal,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Query failed (HTTP ${response.status})`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      // SSE events are separated by double newlines
      const events = buffer.split('\n\n');
      buffer = events.pop() || '';

      for (const event of events) {
        // Handle both "data: {...}" and "data:  {...}" (extra whitespace)
        const dataMatch = event.match(/^data:\s*(.+)$/ms);
        if (!dataMatch) continue;

        try {
          const data = JSON.parse(dataMatch[1].trim());
          if (data.type === 'thinking') {
            onThinking && onThinking(data.content);
          } else if (data.type === 'error') {
            // Structured error from backend
            onError && onError(data.content || 'An unknown error occurred');
          } else if (data.type === 'payload' || data.data) {
            onPayload && onPayload(data.data || data);
          }
        } catch (e) {
          // Non-JSON line — log and skip, don't crash the stream
          console.warn('Skipping non-JSON SSE chunk:', dataMatch[1]?.substring(0, 100));
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') {
      onError && onError('AI query timed out after 120 seconds. Try a simpler question or check API connectivity.');
    } else {
      onError && onError(err.message);
    }
  } finally {
    clear();
  }
};

// ---------------------------------------------------------------------------
// Fetch Forecast
// ---------------------------------------------------------------------------
export const fetchForecast = async (historicalValues, periods = 4) => {
  const response = await fetch(`${BASE_URL}/forecast`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ historical_values: historicalValues, periods }),
  });
  return await response.json();
};

// ---------------------------------------------------------------------------
// Dataset Preview
// ---------------------------------------------------------------------------
export const fetchDatasetPreview = async (datasetId, page = 1, pageSize = 50) => {
  const response = await fetch(`${BASE_URL}/dataset/${datasetId}/preview?page=${page}&page_size=${pageSize}`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Failed to fetch dataset preview' }));
    throw new Error(err.detail);
  }
  return await response.json();
};

// ---------------------------------------------------------------------------
// Preprocess Dataset
// ---------------------------------------------------------------------------
export const preprocessDataset = async (datasetId, command) => {
  const response = await fetch(`${BASE_URL}/dataset/${datasetId}/preprocess`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ command }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Preprocessing failed' }));
    throw new Error(err.detail);
  }
  return await response.json();
};

// ---------------------------------------------------------------------------
// Download Cleaned Dataset
// ---------------------------------------------------------------------------
export const downloadDataset = async (datasetId, fallbackName = 'cleaned_dataset.csv') => {
  const response = await fetch(`${BASE_URL}/dataset/${datasetId}/download`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Failed to download dataset' }));
    throw new Error(err.detail || 'Failed to download dataset');
  }

  let filename = fallbackName;
  const disposition = response.headers.get('Content-Disposition');
  if (disposition && disposition.includes('filename=')) {
    const match = disposition.match(/filename=["']?([^"']+)["']?/);
    if (match && match[1]) {
      filename = match[1];
    }
  }

  const blob = await response.blob();
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  link.parentNode.removeChild(link);
  window.URL.revokeObjectURL(url);
};

// ---------------------------------------------------------------------------
// Health Check — AI Engine status badge
// ---------------------------------------------------------------------------
export const fetchHealth = async () => {
  try {
    const response = await fetch(`${BASE_URL}/health`);
    if (!response.ok) return { llm_configured: false, llm_reachable: false, model: '' };
    return await response.json();
  } catch {
    return { llm_configured: false, llm_reachable: false, model: '' };
  }
};
