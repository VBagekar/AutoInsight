import React, { useState, useRef, useEffect } from 'react';
import { 
  LayoutDashboard, Database, MessageSquare, LineChart, FileText, Settings, 
  ChevronLeft, Sparkles, Maximize2, X, AlertCircle, TrendingUp, ChevronDown, Upload, Plus, Trash2, Search, Play, Download, Copy, Check, BarChart2,
  Clock, Activity, Cpu, Zap, Loader2
} from 'lucide-react';
import { ChartRenderer, CATEGORIZED_CHARTS, AVAILABLE_CHARTS } from './ChartComponents';
import { uploadDataset, streamAIQuery, fetchDatasetPreview, preprocessDataset, downloadDataset, fetchHealth } from '../api/client';
import './Dashboard.css';

// Chart Chooser Dropdown Component for single cards
const ChartChooser = ({ currentType, onSelect }) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <div className="chart-chooser" ref={dropdownRef}>
      <button className="chooser-btn glass-panel" onClick={() => setIsOpen(!isOpen)}>
        {currentType} <ChevronDown size={14} />
      </button>
      {isOpen && (
        <div className="chooser-menu glass-panel animate-fade-in-up" style={{ animationDuration: '0.2s', maxHeight: '320px', width: '220px' }}>
          {Object.entries(CATEGORIZED_CHARTS).map(([category, charts]) => (
            <div key={category} className="chooser-category">
              <div className="chooser-category-title">{category}</div>
              {charts.map(type => (
                <button 
                  key={type} 
                  className={`chooser-item ${currentType === type ? 'active' : ''}`}
                  onClick={() => {
                    onSelect(type);
                    setIsOpen(false);
                  }}
                >
                  {type}
                </button>
              ))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

// Dynamic Chart Card Component
const ChartCard = ({ chart, isExpanded, onToggleExpand, onRemove, onTypeChange }) => {
  return (
    <div className={`chart-card glass-panel ${isExpanded ? 'expanded' : ''}`}>
      <div className="card-header">
        <div className="card-title-group">
          <h3>{chart.title}</h3>
          <ChartChooser currentType={chart.type} onSelect={(newType) => onTypeChange(chart.id, newType)} />
        </div>
        <div style={{ display: 'flex', gap: '6px' }}>
          <button className="icon-btn small" onClick={() => onToggleExpand(chart.id)} title={isExpanded ? "Collapse" : "Expand"}>
            {isExpanded ? <X size={16}/> : <Maximize2 size={16}/>}
          </button>
          <button className="icon-btn small" onClick={() => onRemove(chart.id)} title="Remove Chart">
            <Trash2 size={15} style={{ color: 'var(--c-accent-orange)' }} />
          </button>
        </div>
      </div>
      
      <div className="chart-container">
        <ChartRenderer 
          type={chart.type} 
          data={chart.data} 
          xAxis={chart.x_axis} 
          yAxis={chart.y_axis} 
          secondaryDimension={chart.secondary_dimension}
          matrixData={chart.matrix_data}
        />
      </div>
      
      {chart.insight_tooltip && (
        <div className="tooltip-insight" style={{ marginTop: '12px' }}>
          <Sparkles size={14} color="var(--c-accent-purple)" style={{ marginTop: '2px', flexShrink: 0 }} />
          <span>{chart.insight_tooltip}</span>
        </div>
      )}

      {isExpanded && (
        <div className="expanded-insights animate-fade-in-up">
          <h4>Analytical Deep Dive ({chart.type})</h4>
          <p>
            This visualization analyzes <strong>{chart.y_axis || 'Metric'}</strong> aggregated across <strong>{chart.x_axis || 'Dimension'}</strong>.
            Hover over visual elements to inspect interactive PowerBI data cards, percentage contributions, and statistical distributions.
          </p>
          <div className="insight-tags">
            <span className="tag success"><TrendingUp size={12}/> Verified Computation</span>
            <span className="tag warning"><AlertCircle size={12}/> Cleaned & Validated</span>
          </div>
        </div>
      )}
    </div>
  );
};

// Add Chart Modal Component
const AddChartModal = ({ isOpen, onClose, onAddChart }) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [activeCategory, setActiveCategory] = useState('All');

  if (!isOpen) return null;

  const categories = ['All', ...Object.keys(CATEGORIZED_CHARTS)];

  const getFilteredCharts = () => {
    let list = [];
    if (activeCategory === 'All') {
      list = AVAILABLE_CHARTS;
    } else {
      list = CATEGORIZED_CHARTS[activeCategory] || [];
    }

    if (searchTerm.trim()) {
      list = list.filter(c => c.toLowerCase().includes(searchTerm.toLowerCase()));
    }
    return list;
  };

  return (
    <div className="modal-backdrop glass-panel" style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, zIndex: 1000,
      background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(8px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px'
    }}>
      <div className="glass-panel" style={{
        width: '100%', maxWidth: '640px', maxHeight: '80vh', display: 'flex', flexDirection: 'column',
        borderRadius: '24px', padding: '24px', background: 'var(--c-bg)', border: '1px solid var(--c-glass-border)'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '1.25rem', fontWeight: 600 }}>Select Visualization Type</h3>
          <button className="icon-btn small" onClick={onClose}><X size={18}/></button>
        </div>

        {/* Search & Category Filter */}
        <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <Search size={16} style={{ position: 'absolute', left: '12px', top: '12px', color: 'var(--c-text-secondary)' }} />
            <input 
              type="text"
              placeholder="Search visualization catalog..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              style={{
                width: '100%', padding: '10px 12px 10px 36px', borderRadius: '12px',
                border: '1px solid var(--c-glass-border)', background: 'var(--c-glass-bg)',
                color: 'var(--c-text)', outline: 'none'
              }}
            />
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px', overflowX: 'auto', paddingBottom: '12px', marginBottom: '12px' }}>
          {categories.map(cat => (
            <button 
              key={cat} 
              className={`btn ${activeCategory === cat ? 'btn-primary' : 'btn-glass'}`}
              style={{ padding: '6px 14px', fontSize: '0.82rem', whiteSpace: 'nowrap' }}
              onClick={() => setActiveCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Chart List Grid */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(170px, 1fr))', gap: '10px', paddingRight: '4px' }}>
          {getFilteredCharts().map(type => (
            <button 
              key={type}
              className="glass-panel"
              style={{
                padding: '12px', borderRadius: '12px', border: '1px solid var(--c-glass-border)',
                textAlign: 'left', background: 'var(--c-glass-bg)', cursor: 'pointer',
                transition: 'all 0.2s', display: 'flex', flexDirection: 'column', gap: '4px'
              }}
              onClick={() => {
                onAddChart(type);
                onClose();
              }}
            >
              <span style={{ fontWeight: 600, fontSize: '0.9rem', color: 'var(--c-text)' }}>{type}</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--c-text-secondary)' }}>Click to materialize</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

// Executive Report Modal Component with Markdown & Export
const ReportModal = ({ isOpen, onClose, report, datasetName }) => {
  const [copied, setCopied] = useState(false);
  if (!isOpen) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(report || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownload = () => {
    const blob = new Blob([report || ''], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${datasetName || 'executive'}_insights_report.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="modal-backdrop glass-panel" style={{
      position: 'fixed', inset: 0, zIndex: 1000,
      background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(10px)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '24px'
    }}>
      <section className="glass-panel animate-fade-in-up" style={{
        width: 'min(820px, 100%)', maxHeight: '85vh', overflow: 'hidden',
        borderRadius: '24px', padding: '28px', background: 'var(--c-bg)',
        border: '1px solid var(--c-glass-border)', display: 'flex', flexDirection: 'column'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px', borderBottom: '1px solid var(--c-glass-border)', paddingBottom: '12px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <FileText size={20} color="var(--c-accent-blue)" />
            <h3 style={{ fontSize: '1.25rem', fontWeight: 700 }}>Executive Analysis Report</h3>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button className="btn btn-glass" onClick={handleCopy} style={{ padding: '6px 12px', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
              {copied ? <Check size={14} color="var(--c-accent-emerald)" /> : <Copy size={14} />}
              {copied ? 'Copied' : 'Copy'}
            </button>
            <button className="btn btn-glass" onClick={handleDownload} style={{ padding: '6px 12px', fontSize: '0.82rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Download size={14} /> Download MD
            </button>
            <button className="icon-btn small" onClick={onClose}><X size={18} /></button>
          </div>
        </div>

        <div style={{
          flex: 1, overflowY: 'auto', whiteSpace: 'pre-wrap', lineHeight: 1.7,
          color: 'var(--c-text)', fontSize: '0.92rem', paddingRight: '8px',
          fontFamily: 'system-ui, -apple-system, sans-serif'
        }}>
          {report || 'Upload a dataset and ask analytical questions to generate an executive report.'}
        </div>
      </section>
    </div>
  );
};

// Main Dashboard Component
const Dashboard = ({ onBack, toggleTheme, theme }) => {
  const [expandedCard, setExpandedCard] = useState(null);
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([
    { 
      role: 'ai', 
      text: 'Welcome to AutoInsight! I am your AI Data Scientist powered by NVIDIA Nemotron-3 Ultra 550B. Upload any CSV/Excel file or ask any question to generate interactive dashboards, automated EDA, and executive insights.' 
    }
  ]);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isReportOpen, setIsReportOpen] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [activeReasoningText, setActiveReasoningText] = useState('');

  const [dashboardTitle, setDashboardTitle] = useState('Upload a dataset to begin');
  const [activeCharts, setActiveCharts] = useState([]);

  const [kpiSummary, setKpiSummary] = useState({
    primary_kpi: '—', value: '—', data_quality: '—', total_rows: '—'
  });

  const [recommendations, setRecommendations] = useState([
    'Upload a CSV or Excel file to receive automated cleaning and data-backed visual recommendations.'
  ]);

  const [datasetSummary, setDatasetSummary] = useState(null);
  const [datasetId, setDatasetId] = useState(null);
  const [detailedReport, setDetailedReport] = useState('');
  const [forecast, setForecast] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [activeView, setActiveView] = useState('dashboard');
  const [previewData, setPreviewData] = useState({ columns: [], rows: [], total_rows: 0, page: 1, page_size: 50, total_pages: 0, cleaning_summary: null });
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState(null);
  const [currentFileName, setCurrentFileName] = useState('');
  const [isDownloading, setIsDownloading] = useState(false);
  const [planSource, setPlanSource] = useState(null);
  const [cleaningExpanded, setCleaningExpanded] = useState(false);

  const chatEndRef = useRef(null);

  useEffect(() => {
    let interval = null;
    if (isThinking || isUploading || isDownloading) {
      setElapsedTime(0);
      interval = setInterval(() => {
        setElapsedTime(prev => +(prev + 0.1).toFixed(1));
      }, 100);
    } else {
      clearInterval(interval);
    }
    return () => clearInterval(interval);
  }, [isThinking, isUploading, isDownloading]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [chatHistory, activeReasoningText]);

  const handleToggleExpand = (id) => {
    setExpandedCard(expandedCard === id ? null : id);
  };

  const handleRemoveChart = (id) => {
    setActiveCharts(prev => prev.filter(c => c.id !== id));
  };

  const handleTypeChange = (id, newType) => {
    setActiveCharts(prev => prev.map(c => c.id === id ? { ...c, type: newType } : c));
  };

  const handleAddChart = (chartType) => {
    const newId = `custom-${Date.now()}`;
    const baseChart = activeCharts[0];
    setActiveCharts(prev => [
      ...prev,
      {
        id: newId,
        title: `${chartType} Analysis`,
        type: chartType,
        x_axis: baseChart?.x_axis,
        y_axis: baseChart?.y_axis,
        secondary_dimension: baseChart?.secondary_dimension,
        data: baseChart?.data || [],
        insight_tooltip: `Dynamic visual representation in a ${chartType.toLowerCase()} view.`
      }
    ]);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const res = await uploadDataset(file);
      processDatasetResult(res, file.name);
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'ai', text: `Upload Error: ${err.message}. Check backend connection.` }]);
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadSampleData = async () => {
    setIsUploading(true);
    try {
      const resp = await fetch('/sample_sales.csv');
      const text = await resp.text();
      const blob = new Blob([text], { type: 'text/csv' });
      const file = new File([blob], 'sample_sales.csv', { type: 'text/csv' });

      const res = await uploadDataset(file);
      processDatasetResult(res, 'sample_sales.csv');
    } catch (err) {
      setChatHistory(prev => [...prev, { role: 'ai', text: `Sample data loading error: ${err.message}` }]);
    } finally {
      setIsUploading(false);
    }
  };

  const processDatasetResult = (res, filename) => {
    if (!res || !res.summary) return;
    setCurrentFileName(filename || res.summary?.filename || 'dataset');
    setDatasetSummary(res.summary);
    setDatasetId(res.dataset_id);
    setForecast(res.forecast || null);
    if (res.plan_source) setPlanSource(res.plan_source);
    
    setDashboardTitle(res.dashboard_title || `${res.summary.primary_kpi || 'Domain'} Analytics Dashboard`);
    
    const kpi = res.kpi_summary?.primary_kpi || res.summary.primary_kpi || 'Primary Metric';
    setKpiSummary({
      primary_kpi: kpi,
      value: res.kpi_summary?.formatted_value || (typeof res.kpi_summary?.value === 'number' ? res.kpi_summary.value.toLocaleString() : '—'),
      secondary_kpi: res.kpi_summary?.secondary_kpi || null,
      secondary_value: res.kpi_summary?.secondary_formatted_value || null,
      data_quality: res.kpi_summary?.data_quality || res.summary.quality_score || 100.0,
      total_rows: (res.kpi_summary?.total_rows || res.summary.row_count || 0).toLocaleString()
    });

    if (res.charts && res.charts.length > 0) {
      setActiveCharts(res.charts.map((c, i) => ({
        id: c.id || `auto-${i}`,
        title: c.title || `${c.type} Overview`,
        type: c.type,
        x_axis: c.x_axis,
        y_axis: c.y_axis,
        secondary_dimension: c.secondary_dimension,
        matrix_data: c.matrix_data,
        data: c.data || [],
        insight_tooltip: c.insight_tooltip || `Automated calculation for ${kpi}.`
      })));
    }

    if (res.ai_insights) {
      setRecommendations(res.ai_insights);
    }

    setDetailedReport(res.ai_insights?.join('\n\n') || 'Initial analysis complete. Ask any analytical query to generate an executive report.');
    setChatHistory(prev => [...prev, { 
      role: 'ai', 
      text: `Dataset "${filename}" ingested & cleansed (${res.summary.row_count.toLocaleString()} rows, ${res.summary.column_count} dimensions). Data Integrity: ${res.summary.quality_score}%. Visualized ${res.charts?.length || 6} charts!` 
    }]);
  };

  const handleDownloadData = async () => {
    if (!datasetId) return;
    setIsDownloading(true);
    try {
      const fallback = currentFileName ? `${currentFileName.replace(/\.[^/.]+$/, "")}_cleaned.csv` : 'cleaned_dataset.csv';
      await downloadDataset(datasetId, fallback);
      setChatHistory(prev => [...prev, { 
        role: 'ai', 
        text: '📥 Successfully exported and downloaded cleaned dataset in CSV format.' 
      }]);
    } catch (err) {
      setChatHistory(prev => [...prev, { 
        role: 'ai', 
        text: `Download Error: ${err.message}`, 
        type: 'error' 
      }]);
    } finally {
      setIsDownloading(false);
    }
  };

  const preprocessKeywords = [
    "add a column","add column","drop column","remove column","rename column","fill missing","fill in missing","change type","convert column","filter rows","filter out"
  ];

  const isPreprocessCommand = (msg) => {
    const lower = msg.toLowerCase();
    return preprocessKeywords.some(kw => lower.includes(kw));
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    
    const userMsg = chatInput;
    setChatInput('');
    setChatHistory(prev => [...prev, { role: 'user', text: userMsg }]);

    if (!datasetId) {
      setChatHistory(prev => [...prev, { role: 'ai', text: 'Please upload a CSV/Excel dataset or load the sample data first to run analytical queries.' }]);
      return;
    }

    if (isPreprocessCommand(userMsg)) {
      setChatHistory(prev => [...prev, { role: 'ai', text: '⚙️ Executing dataset preprocessing...' }]);
      try {
        const res = await preprocessDataset(datasetId, userMsg);
        setChatHistory(prev => {
          const updated = [...prev];
          if (updated[updated.length - 1]?.role === 'ai') {
            updated[updated.length - 1] = { role: 'ai', text: res.confirmation_message || 'Preprocessing completed.' };
          }
          return updated;
        });
        if (activeView === 'datasets') {
          await fetchPreview(previewData.page);
        }
      } catch (err) {
        setChatHistory(prev => {
          const updated = [...prev];
          if (updated[updated.length - 1]?.role === 'ai') {
            updated[updated.length - 1] = { role: 'ai', text: err.message || 'Preprocessing failed.', type: 'error' };
          }
          return updated;
        });
      }
      return;
    }

    // Normal analytical query path with live reasoning
    setIsThinking(true);
    setActiveReasoningText('');

    let accumulatedReasoning = '';

    await streamAIQuery(
      userMsg,
      datasetId,
      (thinkingChunk) => {
        accumulatedReasoning += thinkingChunk;
        setActiveReasoningText(accumulatedReasoning);
      },
      (payload) => {
        setIsThinking(false);
        setActiveReasoningText('');
        if (!payload) return;

        if (payload.dashboard_title) {
          setDashboardTitle(payload.dashboard_title);
        }
        if (payload.suggested_charts && payload.suggested_charts.length > 0) {
          setActiveCharts(payload.suggested_charts.map((c, i) => ({
            id: c.id || `stream-${i}`,
            title: c.title || `${c.type} Chart`,
            type: c.type,
            x_axis: c.x_axis,
            y_axis: c.y_axis,
            secondary_dimension: c.secondary_dimension,
            matrix_data: c.matrix_data,
            data: c.data || [],
            insight_tooltip: c.insight_tooltip || 'Verified calculation from cleaned dataset.'
          })));
        }
        if (payload.ai_recommendations) {
          setRecommendations(payload.ai_recommendations);
        }
        if (payload.kpi_summary) {
          setKpiSummary(prev => ({
            ...prev,
            primary_kpi: payload.kpi_summary.primary_kpi || prev.primary_kpi,
            value: payload.kpi_summary.formatted_value || (typeof payload.kpi_summary.value === 'number' ? payload.kpi_summary.value.toLocaleString() : prev.value),
            total_rows: payload.kpi_summary.total_rows?.toLocaleString?.() || prev.total_rows,
            data_quality: payload.kpi_summary.data_quality ?? prev.data_quality
          }));
        }
        if (payload.detailed_report) setDetailedReport(payload.detailed_report);
        if (payload.forecast) setForecast(payload.forecast);
        if (payload.plan_source) setPlanSource(payload.plan_source);

        const sourceLabel = payload.plan_source === 'llm' ? '✨ Nemotron-3 Ultra 550B' : '⚙️ Rule-based fallback';

        setChatHistory(prev => [
          ...prev,
          {
            role: 'ai',
            text: `✨ Dashboard updated with ${payload.suggested_charts?.length || 6} visualizations for: "${userMsg}".`,
            sourceLabel: sourceLabel,
            fullReasoning: accumulatedReasoning,
          }
        ]);
      },
      (error) => {
        setIsThinking(false);
        setActiveReasoningText('');
        setChatHistory(prev => [
          ...prev,
          {
            role: 'ai',
            text: `Query Error: ${error}`,
            type: 'error'
          }
        ]);
      }
    );
  };

  const handlePromptClick = (prompt) => {
    setChatInput(prompt);
  };

  const fetchPreview = async (pageNum = 1) => {
    if (!datasetId) return;
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const data = await fetchDatasetPreview(datasetId, pageNum, 50);
      setPreviewData(data);
    } catch (err) {
      setPreviewError(err.message || 'Failed to load preview');
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleViewChange = (view) => {
    setActiveView(view);
    if (view === 'datasets' && datasetId) {
      fetchPreview(1);
    }
  };

  return (
    <div className="dashboard-layout">
      {(isThinking || isUploading || isDownloading) && <div className="top-shimmer-bar" />}

      <AddChartModal 
        isOpen={isAddModalOpen} 
        onClose={() => setIsAddModalOpen(false)} 
        onAddChart={handleAddChart} 
      />
      <ReportModal 
        isOpen={isReportOpen} 
        onClose={() => setIsReportOpen(false)} 
        report={detailedReport} 
        datasetName={currentFileName}
      />

      {/* Left Sidebar */}
      <aside className="sidebar glass-panel">
        <div className="sidebar-top">
          <button className="icon-btn" onClick={onBack} title="Back to Home">
            <ChevronLeft size={20} />
          </button>
          <div className="divider"></div>
          <button className={`icon-btn ${activeView === 'dashboard' ? 'active' : ''}`} title="Dashboard" onClick={() => handleViewChange('dashboard')}>
            <LayoutDashboard size={20} />
          </button>
          <button className={`icon-btn ${activeView === 'datasets' ? 'active' : ''}`} title="Datasets Preview" onClick={() => handleViewChange('datasets')}>
            <Database size={20} />
          </button>
        </div>
        <div className="sidebar-bottom">
          <button className="icon-btn" title="Toggle Theme" onClick={toggleTheme}>
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="workspace">
        {activeView === 'dashboard' ? (
          <>
            <header className="workspace-header">
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
                  <h2>{dashboardTitle}</h2>
                  {(isThinking || isUploading) && (
                    <div className="ai-active-pill glass-panel">
                      <div className="ai-pulse-dot" />
                      <span className="ai-active-title">
                        {isUploading ? 'Ingesting Dataset' : 'Nemotron-3 Ultra 550B'}
                      </span>
                      <span className="ai-active-step">
                        {isUploading
                          ? 'Cleansing & profiling schema'
                          : activeReasoningText
                            ? 'Streaming reasoning & aggregations'
                            : 'Analyzing intent & selecting visuals'}
                      </span>
                      <span className="ai-active-timer">
                        <Clock size={12} /> {elapsedTime.toFixed(1)}s
                      </span>
                    </div>
                  )}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--c-text-secondary)', marginTop: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span>{currentFileName || 'No dataset loaded'}</span>
                  {planSource && (
                    <span style={{
                      fontSize: '0.72rem', padding: '2px 8px', borderRadius: '999px',
                      background: planSource === 'llm' ? 'rgba(139,92,246,0.15)' : 'rgba(245,158,11,0.15)',
                      color: planSource === 'llm' ? 'var(--c-accent-purple)' : '#f59e0b',
                      border: `1px solid ${planSource === 'llm' ? 'rgba(139,92,246,0.3)' : 'rgba(245,158,11,0.3)'}`,
                      fontWeight: 600,
                    }}>
                      {planSource === 'llm' ? '✨ Nemotron-3 Ultra 550B' : '⚙️ Rule-based fallback'}
                    </span>
                  )}
                </div>
              </div>

              <div className="header-actions">
                <button className="btn btn-glass" onClick={handleLoadSampleData} style={{ gap: '6px' }} disabled={isUploading || isThinking}>
                  <Play size={15} color="var(--c-accent-emerald)" /> Sample CSV
                </button>

                <label className="btn btn-glass" style={{ cursor: isUploading || isThinking ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', gap: '6px', opacity: isUploading || isThinking ? 0.7 : 1 }}>
                  <Upload size={16} /> {isUploading ? 'Uploading...' : 'Upload CSV/Excel'}
                  <input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileUpload} style={{ display: 'none' }} disabled={isUploading || isThinking} />
                </label>

                <button className="btn btn-glass" onClick={() => setIsReportOpen(true)} style={{ gap: '6px' }} disabled={!datasetId}>
                  <FileText size={16} /> Executive Report
                </button>

                <button className="btn btn-primary" onClick={() => setIsAddModalOpen(true)} style={{ gap: '6px' }} disabled={!datasetId}>
                  <Plus size={16} /> Add Chart
                </button>
              </div>
            </header>

            {/* Top KPI Cards Row */}
            <div style={{
              display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
              gap: '16px', marginBottom: '20px'
            }}>
              <div className="glass-panel" style={{ padding: '16px', borderRadius: '16px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--c-text-secondary)', fontWeight: 600 }}>
                  PRIMARY KPI ({kpiSummary.primary_kpi})
                </span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '4px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--c-accent-blue)' }}>
                  {kpiSummary.value}
                </div>
              </div>

              {kpiSummary.secondary_kpi ? (
                <div className="glass-panel" style={{ padding: '16px', borderRadius: '16px' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--c-text-secondary)', fontWeight: 600 }}>
                    SECONDARY KPI ({kpiSummary.secondary_kpi})
                  </span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '4px', color: 'var(--c-accent-emerald)' }}>
                    {kpiSummary.secondary_value || '—'}
                  </div>
                </div>
              ) : (
                <div className="glass-panel" style={{ padding: '16px', borderRadius: '16px' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--c-text-secondary)', fontWeight: 600 }}>CLEANED RECORDS</span>
                  <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '4px' }}>
                    {kpiSummary.total_rows}
                  </div>
                </div>
              )}

              <div className="glass-panel" style={{ padding: '16px', borderRadius: '16px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--c-text-secondary)', fontWeight: 600 }}>
                  {kpiSummary.secondary_kpi ? 'CLEANED RECORDS' : 'DATA QUALITY SCORE'}
                </span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '4px', color: kpiSummary.secondary_kpi ? 'var(--c-text)' : 'var(--c-accent-emerald)' }}>
                  {kpiSummary.secondary_kpi ? kpiSummary.total_rows : `${kpiSummary.data_quality}%`}
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '16px', borderRadius: '16px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--c-text-secondary)', fontWeight: 600 }}>
                  {forecast?.projected_growth_rate ? 'PROJECTED GROWTH RATE' : 'DATA INTEGRITY'}
                </span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '4px', color: 'var(--c-accent-purple)' }}>
                  {forecast?.projected_growth_rate || `${kpiSummary.data_quality}%`}
                </div>
              </div>
            </div>

            {/* Canvas & Charts */}
            <div className="canvas">
              {isThinking && (
                <div className="canvas-processing-banner glass-panel">
                  <div className="banner-left">
                    <Sparkles size={18} color="var(--c-accent-purple)" className="animate-spin-slow" />
                    <div>
                      <div className="banner-text">NVIDIA Nemotron-3 Ultra 550B is formulating visual intelligence...</div>
                      <div className="banner-subtext">
                        {activeReasoningText 
                          ? 'Streaming live thoughts, verifying column axes, and calculating distributions...' 
                          : 'Parsing prompt intent and selecting optimal chart compositions...'}
                      </div>
                    </div>
                  </div>
                  <div className="ai-active-timer">
                    <Clock size={12} /> {elapsedTime.toFixed(1)}s
                  </div>
                </div>
              )}

              {activeCharts.map((chart) => (
                <ChartCard 
                  key={chart.id}
                  chart={chart}
                  isExpanded={expandedCard === chart.id} 
                  onToggleExpand={handleToggleExpand} 
                  onRemove={handleRemoveChart}
                  onTypeChange={handleTypeChange}
                />
              ))}

              {!activeCharts.length && !isUploading && (
                <div className="recommendation-card liquid-glass" style={{ gridColumn: '1 / -1', padding: '32px', textAlign: 'center' }}>
                  <Database size={36} color="var(--c-accent-purple)" style={{ margin: '0 auto 12px' }} />
                  <h3 style={{ fontSize: '1.2rem', marginBottom: '8px' }}>Ready to Analyze Your Data</h3>
                  <p style={{ fontSize: '0.9rem', color: 'var(--c-text-secondary)', maxWidth: '480px', margin: '0 auto' }}>
                    Upload any CSV or Excel file, or click "Sample CSV" above. The application will automatically execute automated EDA, data cleaning, and Nemotron-powered dashboard visual generation.
                  </p>
                </div>
              )}
              
              {/* Recommendation Card */}
              {activeCharts.length > 0 && (
                <div className="recommendation-card liquid-glass">
                  <div className="rec-header">
                    <Sparkles size={18} color="var(--c-accent-purple)" />
                    <h3>Automated Insights & Recommendations</h3>
                  </div>
                  <ul style={{ paddingLeft: '18px', fontSize: '0.88rem', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                  <button className="btn btn-primary" style={{marginTop: '16px', width: '100%'}} onClick={() => setIsReportOpen(true)}>
                    View Executive Report
                  </button>
                </div>
              )}
            </div>
          </>
        ) : (
          <div style={{ height: 'calc(100% - 20px)', display: 'flex', flexDirection: 'column' }}>
            <header className="workspace-header" style={{ flexShrink: 0 }}>
              <div>
                <h2>Dataset Preview & Preprocessing</h2>
                <div style={{ fontSize: '0.85rem', color: 'var(--c-text-secondary)', marginTop: '2px' }}>
                  Cleaned & preprocessed table view {currentFileName ? `• ${currentFileName}` : ''}
                </div>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <button 
                  className="btn btn-primary"
                  onClick={handleDownloadData}
                  disabled={!datasetId || isDownloading}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '8px' }}
                  title="Download full preprocessed dataset in CSV format"
                >
                  <Download size={16} />
                  {isDownloading ? 'Downloading...' : 'Download Cleaned CSV'}
                </button>
              </div>
            </header>
            
            <div className="glass-panel" style={{ borderRadius: '16px', overflow: 'hidden', flex: 1, display: 'flex', flexDirection: 'column' }}>
              {previewLoading ? (
                <div style={{ padding: '40px', textAlign: 'center', color: 'var(--c-text-secondary)' }}>Loading records...</div>
              ) : previewError ? (
                <div style={{ padding: '24px', color: 'var(--c-accent-orange)' }}>Error: {previewError}</div>
              ) : previewData.columns.length > 0 ? (
                <>
                  <div style={{ overflowX: 'auto', flex: 1 }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                      <thead style={{ position: 'sticky', top: 0, zIndex: 1 }}>
                        <tr style={{ background: 'var(--c-glass-bg)', borderBottom: '1px solid var(--c-glass-border)' }}>
                          {previewData.columns.map(col => (
                            <th key={col} style={{ padding: '10px 14px', textAlign: 'left', fontWeight: 600, color: 'var(--c-text)', whiteSpace: 'nowrap' }}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {previewData.rows.map((row, i) => (
                          <tr key={i} style={{ borderBottom: '1px solid var(--c-glass-border)', background: i % 2 === 0 ? 'transparent' : 'var(--c-glass-bg)' }}>
                            {previewData.columns.map(col => (
                              <td key={col} style={{ padding: '10px 14px', color: 'var(--c-text)', whiteSpace: 'nowrap' }}>
                                {row[col] === null || row[col] === undefined ? <span style={{ color: 'var(--c-text-muted)' }}>-</span> : String(row[col])}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <div style={{ padding: '12px 16px', borderTop: '1px solid var(--c-glass-border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
                    <div style={{ fontSize: '0.85rem', color: 'var(--c-text-secondary)' }}>
                      Page {previewData.page} of {previewData.total_pages || 1} — {previewData.total_rows.toLocaleString()} total rows
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <button className="btn btn-glass" disabled={previewData.page <= 1} onClick={() => fetchPreview(previewData.page - 1)}>Previous</button>
                      <button className="btn btn-glass" disabled={previewData.page >= previewData.total_pages} onClick={() => fetchPreview(previewData.page + 1)}>Next</button>
                    </div>
                  </div>
                </>
              ) : (
                <div style={{ padding: '40px', textAlign: 'center', color: 'var(--c-text-secondary)' }}>No dataset loaded. Upload a file to preview.</div>
              )}
            </div>
          </div>
        )}
      </main>

      {/* Right AI Copilot */}
      <aside className="copilot-panel glass-panel">
        <div className="copilot-header">
          <div className="ai-avatar">
            <Sparkles size={16} />
          </div>
          <div>
            <h3>AutoInsights Copilot</h3>
            <span style={{ fontSize: '0.75rem', color: 'var(--c-accent-emerald)', fontWeight: 600 }}>● Nemotron-3 Ultra 550B</span>
          </div>
        </div>
        
        <div className="chat-history">
          {chatHistory.map((msg, idx) => (
            <div key={idx} className={`chat-bubble ${msg.role}`} style={msg.type === 'error' ? { borderLeft: '4px solid var(--c-accent-orange)', background: 'rgba(255,165,0,0.15)' } : {}}>
              <div style={{ whiteSpace: 'pre-wrap' }}>{msg.text}</div>
              {msg.role === 'ai' && msg.sourceLabel && (
                <div style={{ marginTop: '6px' }}>
                  <span style={{
                    fontSize: '0.7rem', padding: '2px 7px', borderRadius: '999px',
                    background: 'rgba(139,92,246,0.15)',
                    color: 'var(--c-accent-purple)',
                    border: '1px solid rgba(139,92,246,0.3)',
                    fontWeight: 600,
                  }}>
                    {msg.sourceLabel}
                  </span>
                </div>
              )}
            </div>
          ))}

          {/* Real-Time Live Reasoning Box when isThinking is active */}
          {isThinking && (
            <div className="live-reasoning-card animate-fade-in-up">
              <div className="reasoning-header">
                <div className="reasoning-title-group">
                  <div className="ai-pulse-dot" />
                  <span>Nemotron-3 Ultra 550B Active</span>
                </div>
                <span className="ai-active-timer">
                  <Clock size={11} /> {elapsedTime.toFixed(1)}s
                </span>
              </div>
              <div className="terminal-reasoning-stream">
                {activeReasoningText || 'Connecting to NVIDIA Ultra 550B inference gateway & synthesizing schema...'}
                <span className="cursor-blink" />
              </div>
              <div className="step-chips">
                <span className="step-chip done"><Check size={10} /> Schema Ready</span>
                <span className={`step-chip ${activeReasoningText ? 'active' : 'pending'}`}>
                  <Activity size={10} /> {activeReasoningText ? 'Ultra 550B Reasoning' : 'Synthesizing'}
                </span>
                <span className="step-chip pending">
                  <BarChart2 size={10} /> Materializing Visuals
                </span>
              </div>
            </div>
          )}
          <div ref={chatEndRef} />
        </div>

        <div className="suggested-prompts">
          <button onClick={() => handlePromptClick("i want the insights of sales for the current year with respect to everything in the dataset")}>
            "Insights of sales for current year with respect to everything"
          </button>
          <button onClick={() => handlePromptClick("Compare sales and profit across regions with stacked category breakdown")}>
            "Compare sales & profit by region with stacked breakdown"
          </button>
          <button onClick={() => handlePromptClick("Forecast next 4 quarters with 95% confidence intervals")}>
            "Forecast next 4 quarters with 95% confidence intervals"
          </button>
        </div>

        <form className="chat-input-area" onSubmit={handleChatSubmit}>
          <input 
            type="text" 
            placeholder="Ask any question about your data..." 
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            disabled={isThinking}
          />
          <button type="submit" className="send-btn" disabled={isThinking}>
            <Sparkles size={16} />
          </button>
        </form>
      </aside>
    </div>
  );
};

export default Dashboard;
