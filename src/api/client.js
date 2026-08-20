const BASE_URL = 'http://localhost:8000/api';

export const uploadDataset = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    throw new Error('Failed to upload dataset');
  }

  return await response.json();
};

export const streamAIQuery = async (query, datasetId, onThinking, onPayload, onError) => {
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
    });

    if (!response.ok) {
      throw new Error('Failed to query AI engine');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            if (data.type === 'thinking') {
              onThinking && onThinking(data.content);
            } else if (data.type === 'payload' || data.data) {
              onPayload && onPayload(data.data || data);
            }
          } catch (e) {
            console.error('Error parsing SSE line:', e);
          }
        }
      }
    }
  } catch (err) {
    onError && onError(err.message);
  }
};

export const fetchForecast = async (historicalValues, periods = 4) => {
  const response = await fetch(`${BASE_URL}/forecast`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ historical_values: historicalValues, periods }),
  });
  return await response.json();
};

export const fetchDatasetPreview = async (datasetId, page = 1, pageSize = 50) => {
  const response = await fetch(`${BASE_URL}/dataset/${datasetId}/preview?page=${page}&page_size=${pageSize}`);
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Failed to fetch dataset preview' }));
    throw new Error(err.detail);
  }
  return await response.json();
};

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

