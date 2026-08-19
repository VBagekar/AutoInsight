import React, { useState, useRef, useEffect } from 'react';
import { 
  LayoutDashboard, Database, MessageSquare, LineChart, FileText, Settings, 
  ChevronLeft, Sparkles, Maximize2, X, AlertCircle, TrendingUp, ChevronDown, Upload, Plus, Trash2, Search, Play
} from 'lucide-react';
import { ChartRenderer, CATEGORIZED_CHARTS, AVAILABLE_CHARTS } from './ChartComponents';
import { uploadDataset, streamAIQuery, fetchDatasetPreview } from '../api/client';
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
        <ChartRenderer type={chart.type} data={chart.data} xAxis={chart.x_axis} yAxis={chart.y_axis} />
      </div>
      
      {chart.insight_tooltip && (
        <div className="tooltip-insight" style={{ marginTop: '12px' }}>
          <Sparkles size={14} color="var(--c-accent-purple)" style={{ marginTop: '2px', flexShrink: 0 }} />
          <span>{chart.insight_tooltip}</span>
        </div>
      )}

      {isExpanded && (
        <div className="expanded-insights animate-fade-in-up">
          <h4>AI Deep Analysis for {chart.type}</h4>
          <p>This {chart.type.toLowerCase()} visualization highlights key operational metrics and statistical variances in your dataset. High concentration patterns indicate opportunities for resource reallocation.</p>
          <div className="insight-tags">
            <span className="tag success"><TrendingUp size={12}/> Positive Trend</span>
            <span className="tag warning"><AlertCircle size={12}/> Anomaly Detected</span>
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
              placeholder="Search 60+ visualization types..."
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
              <span style={{ fontSize: '0.75rem', color: 'var(--c-text-secondary)' }}>Click to add</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

const ReportModal = ({ isOpen, onClose, report }) => {
  if (!isOpen) return null;
  return (
    <div className="modal-backdrop glass-panel" style={{ position: 'fixed', inset: 0, zIndex: 1000, background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(8px)', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '20px' }}>
      <section className="glass-panel" style={{ width: 'min(760px, 100%)', maxHeight: '80vh', overflow: 'auto', borderRadius: '24px', padding: '24px', background: 'var(--c-bg)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
          <h3 style={{ fontSize: '1.2rem' }}>Detailed analysis report</h3>
          <button className="icon-btn small" onClick={onClose}><X size={18} /></button>
        </div>
        <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.65, color: 'var(--c-text-secondary)' }}>{report || 'Ask a question after uploading data to generate a report.'}</div>
      </section>
    </div>
  );
};

const Dashboard = ({ onBack, toggleTheme, theme }) => {
  const [expandedCard, setExpandedCard] = useState(null);
  const [chatInput, setChatInput] = useState('');
  const [chatHistory, setChatHistory] = useState([
    { role: 'ai', text: 'Hello! I am your AutoInsights AI Data Scientist powered by Nemotron-3. Upload a CSV or ask questions to dynamically customize your dashboard.' }
  ]);
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isReportOpen, setIsReportOpen] = useState(false);

  const [dashboardTitle, setDashboardTitle] = useState('Upload a dataset to begin');
  const [activeCharts, setActiveCharts] = useState([]);

  const [kpiSummary, setKpiSummary] = useState({
    primary_kpi: '—', value: '—', data_quality: '—', total_rows: '—'
  });

  const [recommendations, setRecommendations] = useState([
    'Upload a CSV or Excel file to receive data-backed recommendations.'
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
    setActiveCharts(prev => [
      ...prev,
      {
        id: newId,
        title: `${chartType} Analysis`,
        type: chartType,
        x_axis: activeCharts[0]?.x_axis,
        y_axis: activeCharts[0]?.y_axis,
        data: activeCharts[0]?.data || [],
        insight_tooltip: `Uses the current dataset aggregation in a ${chartType.toLowerCase()} view.`
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
      setChatHistory(prev => [...prev, { role: 'ai', text: `Upload Error: ${err.message}. Check backend server.` }]);
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
    setDatasetSummary(res.summary);
    setDatasetId(res.dataset_id);
    setForecast(res.forecast || null);
    
    const kpi = res.summary.primary_kpi || 'Sales';
    setKpiSummary({
      primary_kpi: kpi,
      value: typeof res.kpi_summary?.value === 'number' ? res.kpi_summary.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—',
      data_quality: res.summary.quality_score || 98.5,
      total_rows: res.summary.row_count.toLocaleString()
    });

    if (res.charts && res.charts.length > 0) {
      setActiveCharts(res.charts.map((c, i) => ({
        id: `auto-${i}`,
        title: c.title || `${c.type} Overview`,
        type: c.type === 'Area' ? 'Area Graph' : c.type === 'Bar' ? 'Bar Chart' : c.type === 'Donut' ? 'Donut Chart' : c.type === 'Scatter' ? 'Scatterplot' : c.type,
        x_axis: c.x_axis,
        y_axis: c.y_axis,
        data: c.data || [],
        insight_tooltip: c.insight_tooltip || `Automated insight for ${kpi}.`
      })));
    }

    setDetailedReport(res.ai_insights?.join('\n\n') || 'Initial analysis complete. Ask a question for a detailed report.');
    setChatHistory(prev => [...prev, { 
      role: 'ai', 
      text: `Successfully ingested and cleaned ${filename} (${res.summary.row_count} rows, ${res.summary.column_count} columns). Data quality score: ${res.summary.quality_score}%. Dashboard updated!` 
    }]);
  };

  const handleChatSubmit = async (e) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    
    const userMsg = chatInput;
    setChatInput('');
    setChatHistory(prev => [...prev, { role: 'user', text: userMsg }]);

    if (!datasetId) {
      setChatHistory(prev => [...prev, { role: 'ai', text: 'Please upload a CSV or Excel dataset first so I can calculate a real answer.' }]);
      return;
    }

    setChatHistory(prev => [...prev, { role: 'ai', text: '🧠 Nemotron-3 Reasoning in progress...' }]);

    let lastThinking = '';

    await streamAIQuery(
      userMsg,
      datasetId,
      (thinkingChunk) => {
        lastThinking += thinkingChunk;
        setChatHistory(prev => {
          const updated = [...prev];
          if (updated[updated.length - 1]?.role === 'ai') {
            updated[updated.length - 1].text = `🧠 Reasoning: ${lastThinking.slice(-200)}`;
          }
          return updated;
        });
      },
      (payload) => {
        if (!payload) return;
        
        // Handle conversational response
        if (payload.type === 'chat' || payload.message) {
          const msg = payload.message || payload.content || "Hello! How can I help you analyze your data today?";
          setChatHistory(prev => {
            const updated = [...prev];
            if (updated[updated.length - 1]?.role === 'ai') {
              updated[updated.length - 1].text = msg;
            }
            return updated;
          });
          return;
        }

        // Handle visualization layout update
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
            data: c.data || [],
            insight_tooltip: c.insight_tooltip || 'Generated from the uploaded dataset.'
          })));
        }
        if (payload.ai_recommendations) {
          setRecommendations(payload.ai_recommendations);
        }
        if (payload.kpi_summary) {
          setKpiSummary(prev => ({
            ...prev,
            primary_kpi: payload.kpi_summary.primary_kpi || prev.primary_kpi,
            value: typeof payload.kpi_summary.value === 'number' ? payload.kpi_summary.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : prev.value,
            total_rows: payload.kpi_summary.total_rows?.toLocaleString?.() || prev.total_rows,
            data_quality: payload.kpi_summary.data_quality ?? prev.data_quality
          }));
        }
        if (payload.detailed_report) setDetailedReport(payload.detailed_report);
        if (payload.forecast) setForecast(payload.forecast);

        setChatHistory(prev => {
          const updated = [...prev];
          if (updated[updated.length - 1]?.role === 'ai') {
            updated[updated.length - 1].text = `✨ Dashboard updated for "${userMsg}" with ${payload.suggested_charts?.length || 4} optimized visualizations.`;
          }
          return updated;
        });
      },
      (err) => {
        setChatHistory(prev => [...prev, { role: 'ai', text: `AI Connection Note: ${err}` }]);
      }
    );
  };

  const handlePromptClick = (promptText) => {
    setChatInput(promptText);
  };

  const fetchPreview = async (page = 1) => {
    if (!datasetId) return;
    setPreviewLoading(true);
    setPreviewError(null);
    try {
      const data = await fetchDatasetPreview(datasetId, page, previewData.page_size);
      setPreviewData(data);
    } catch (err) {
      setPreviewError(err.message);
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

  const DatasetPreviewView = () => {
    const { columns, rows, total_rows, page, page_size, total_pages, cleaning_summary } = previewData;

    const renderCleaningSummary = () => {
      if (!cleaning_summary) return null;
      const { rows_before, rows_after, duplicates_removed, imputation_details, high_missing_flagged, outlier_treatment_details } = cleaning_summary;
      return (
        <div className="glass-panel" style={{ padding: '16px', borderRadius: '12px', marginBottom: '16px' }}>
          <h4 style={{ fontSize: '0.9rem', fontWeight: 600, marginBottom: '12px', color: 'var(--c-text)' }}>Cleaning Summary</h4>
          <ul style={{ fontSize: '0.85rem', color: 'var(--c-text-secondary)', lineHeight: 1.8, paddingLeft: '18px' }}>
            <li>{rows_before} → {rows_after} rows after cleaning ({duplicates_removed} duplicates removed)</li>
            {imputation_details && imputation_details.length > 0 && (
              <li>Imputation: {imputation_details.map(d => `${d.column} (${d.strategy})`).join(', ')}</li>
            )}
            {high_missing_flagged && high_missing_flagged.length > 0 && (
              <li>High missing flagged: {high_missing_flagged.map(c => c.column).join(', ')}</li>
            )}
            {outlier_treatment_details && outlier_treatment_details.length > 0 && (
              <li>Outlier treatment: {outlier_treatment_details.map(d => `${d.column} (${d.method})`).join(', ')}</li>
            )}
          </ul>
        </div>
      );
    };

    if (previewLoading) {
      return (
        <div className="glass-panel" style={{ padding: '40px', textAlign: 'center', color: 'var(--c-text-secondary)' }}>
          Loading…
        </div>
      );
    }

    if (previewError) {
      return (
        <div className="glass-panel" style={{ padding: '20px', borderRadius: '12px', color: 'var(--c-accent-orange)' }}>
          Error: {previewError}
        </div>
      );
    }

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', height: '100%', overflow: 'auto' }}>
        {renderCleaningSummary()}
        <div className="glass-panel" style={{ borderRadius: '12px', overflow: 'hidden', flex: 1, display: 'flex', flexDirection: 'column' }}>
          {columns.length > 0 && (
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
                <thead>
                  <tr style={{ background: 'var(--c-glass-bg)', borderBottom: '1px solid var(--c-glass-border)' }}>
                    {columns.map(col => (
                      <th key={col} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, color: 'var(--c-text)', whiteSpace: 'nowrap' }}>{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid var(--c-glass-border)', background: i % 2 === 0 ? 'transparent' : 'var(--c-glass-bg)' }}>
                      {columns.map(col => (
                        <td key={col} style={{ padding: '10px 12px', color: 'var(--c-text)', whiteSpace: 'nowrap' }}>
                          {row[col] === null || row[col] === undefined ? <span style={{ color: 'var(--c-text-muted)' }}>-</span> : String(row[col])}
                        </td>
                      ))}
                    </tr>
                  ))}
                  {rows.length === 0 && (
                    <tr>
                      <td colSpan={columns.length} style={{ padding: '20px', textAlign: 'center', color: 'var(--c-text-secondary)' }}>No data</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
          <div style={{ padding: '12px 16px', borderTop: '1px solid var(--c-glass-border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
            <div style={{ fontSize: '0.85rem', color: 'var(--c-text-secondary)' }}>
              Page {page} of {total_pages || 1} — {total_rows} rows total
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button 
                className="btn btn-glass"
                disabled={page <= 1}
                onClick={() => fetchPreview(page - 1)}
              >
                Previous
              </button>
              <button 
                className="btn btn-glass"
                disabled={page >= total_pages}
                onClick={() => fetchPreview(page + 1)}
              >
                Next
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="dashboard-layout">
      {/* Add Chart Modal */}
      <AddChartModal 
        isOpen={isAddModalOpen} 
        onClose={() => setIsAddModalOpen(false)} 
        onAddChart={handleAddChart} 
      />
      <ReportModal isOpen={isReportOpen} onClose={() => setIsReportOpen(false)} report={detailedReport} />

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
          <button className={`icon-btn ${activeView === 'datasets' ? 'active' : ''}`} title="Datasets" onClick={() => handleViewChange('datasets')}>
            <Database size={20} />
          </button>
          <button className="icon-btn" title="AI Chat"><MessageSquare size={20} /></button>
          <button className="icon-btn" title="Forecasting"><LineChart size={20} /></button>
          <button className="icon-btn" title="Reports"><FileText size={20} /></button>
        </div>
        <div className="sidebar-bottom">
          <button className="icon-btn" onClick={toggleTheme} title="Toggle Theme">
            <Settings size={20} />
          </button>
        </div>
      </aside>

      {/* Main Workspace */}
      <main className="workspace">
        {activeView === 'dashboard' ? (
          <>
            <header className="workspace-header">
              <div>
                <h2>{dashboardTitle}</h2>
                <div style={{ fontSize: '0.85rem', color: 'var(--c-text-secondary)', marginTop: '2px' }}>
                  Powered by NVIDIA Nemotron-3 Super 120B & Multi-Agent Engine
                </div>
              </div>

              <div className="header-actions">
                <button className="btn btn-glass" onClick={handleLoadSampleData} style={{ gap: '6px' }} disabled={isUploading}>
                  <Play size={15} color="var(--c-accent-emerald)" /> Sample CSV
                </button>

                <label className="btn btn-glass" style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Upload size={16} /> {isUploading ? 'Uploading...' : 'Upload CSV'}
                  <input type="file" accept=".csv,.xlsx,.xls" onChange={handleFileUpload} style={{ display: 'none' }} />
                </label>

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
                <span style={{ fontSize: '0.8rem', color: 'var(--c-text-secondary)', fontWeight: 600 }}>PRIMARY KPI ({kpiSummary.primary_kpi})</span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '4px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  {kpiSummary.value}
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '16px', borderRadius: '16px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--c-text-secondary)', fontWeight: 600 }}>TOTAL DATA RECORDS</span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '4px' }}>
                  {kpiSummary.total_rows}
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '16px', borderRadius: '16px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--c-text-secondary)', fontWeight: 600 }}>DATA INTEGRITY SCORE</span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '4px', color: 'var(--c-accent-emerald)' }}>
                  {kpiSummary.data_quality}%
                </div>
              </div>

              <div className="glass-panel" style={{ padding: '16px', borderRadius: '16px' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--c-text-secondary)', fontWeight: 600 }}>PROJECTED 90-DAY GROWTH</span>
                <div style={{ fontSize: '1.5rem', fontWeight: 700, marginTop: '4px', color: 'var(--c-accent-purple)' }}>
                  {forecast?.projected_growth_rate || '—'}
                </div>
              </div>
            </div>

            {/* Canvas & Charts */}
            <div className="canvas">
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
                <div className="recommendation-card liquid-glass">
                  <div className="rec-header"><Database size={18} color="var(--c-accent-purple)" /><h3>Ready for your data</h3></div>
                  <p style={{ fontSize: '0.9rem' }}>Upload a CSV or Excel dataset. The dashboard will clean it and render real charts from it.</p>
                </div>
              )}
              
              {/* Recommendation Card */}
              <div className="recommendation-card liquid-glass">
                <div className="rec-header">
                  <Sparkles size={18} color="var(--c-accent-purple)" />
                  <h3>AI Recommendations</h3>
                </div>
                <ul style={{ paddingLeft: '18px', fontSize: '0.88rem', display: 'flex', flexDirection: 'column', gap: '8px' }}>
                  {recommendations.map((rec, i) => (
                    <li key={i}>{rec}</li>
                  ))}
                </ul>
                <button className="btn btn-primary" style={{marginTop: '16px', width: '100%'}} onClick={() => setIsReportOpen(true)}>View detailed report</button>
              </div>
            </div>
          </>
        ) : (
          <div style={{ height: 'calc(100% - 80px)', display: 'flex', flexDirection: 'column' }}>
            <header className="workspace-header" style={{ flexShrink: 0 }}>
              <div>
                <h2>Dataset Preview</h2>
                <div style={{ fontSize: '0.85rem', color: 'var(--c-text-secondary)', marginTop: '2px' }}>
                  Cleaned & preprocessed data view
                </div>
              </div>
            </header>
            <DatasetPreviewView />
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
            <span style={{ fontSize: '0.75rem', color: 'var(--c-accent-emerald)', fontWeight: 600 }}>● Nemotron Agent Online</span>
          </div>
        </div>
        
        <div className="chat-history">
          {chatHistory.map((msg, idx) => (
            <div key={idx} className={`chat-bubble ${msg.role}`}>
              {msg.text}
            </div>
          ))}
        </div>

        <div className="suggested-prompts">
          <button onClick={() => handlePromptClick("Compare sales this year with last year")}>
            "Compare sales this year with last year"
          </button>
          <button onClick={() => handlePromptClick("Forecast next month's revenue with 95% CI")}>
            "Forecast next month's revenue with 95% CI"
          </button>
        </div>

        <form className="chat-input-area" onSubmit={handleChatSubmit}>
          <input 
            type="text" 
            placeholder="Ask anything..." 
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
          />
          <button type="submit" className="send-btn">
            <Sparkles size={16} />
          </button>
        </form>
      </aside>
    </div>
  );
};

export default Dashboard;
