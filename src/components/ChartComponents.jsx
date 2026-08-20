import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Treemap, FunnelChart, Funnel,
  RadialBarChart, RadialBar, Legend
} from 'recharts';
import { timeSeriesData, categoricalData, scatterData, radarData, treemapData, funnelData } from '../data/mockData';

const COLORS = [
  '#3b82f6', '#8b5cf6', '#10b981', '#06b6d4', '#f59e0b',
  '#ec4899', '#6366f1', '#14b8a6', '#f97316', '#84cc16'
];

export const CATEGORIZED_CHARTS = {
  "Trend & Time": [
    "Area Graph", "Line Graph", "Stacked Area Graph"
  ],
  "Comparison & Breakdown": [
    "Bar Chart", "Stacked Bar Graph", "Multi-set Bar Chart", "Donut Chart", "Pie Chart", "Treemap"
  ],
  "Correlation & Distribution": [
    "Scatterplot", "Histogram", "Box Plot", "Radar Chart", "Heatmap"
  ],
  "Pipeline & Performance": [
    "Funnel Chart"
  ]
};

export const AVAILABLE_CHARTS = Object.values(CATEGORIZED_CHARTS).flat();

// PowerBI / Tableau Style Rich Interactive Tooltip
export const PowerBiTooltip = ({ active, payload, label, yAxis, xAxis }) => {
  if (active && payload && payload.length) {
    const mainItem = payload[0];
    const dataItem = mainItem.payload || {};
    const formattedVal = dataItem.formatted_value || (typeof mainItem.value === 'number' ? mainItem.value.toLocaleString() : mainItem.value);
    const pct = dataItem.percentage;
    const nameLabel = label || dataItem.name || (xAxis ? `${xAxis}: ${dataItem[xAxis]}` : mainItem.name);

    return (
      <div className="powerbi-tooltip glass-panel animate-fade-in-up" style={{
        padding: '12px 16px',
        borderRadius: '12px',
        background: 'rgba(15, 23, 42, 0.92)',
        backdropFilter: 'blur(12px)',
        border: '1px solid rgba(255, 255, 255, 0.15)',
        boxShadow: '0 12px 32px rgba(0, 0, 0, 0.4)',
        color: '#f8fafc',
        minWidth: '180px',
        zIndex: 1000
      }}>
        <div style={{ fontSize: '0.8rem', color: '#94a3b8', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: '4px' }}>
          {nameLabel}
        </div>

        {payload.map((entry, index) => {
          const entryVal = entry.value;
          const displayVal = typeof entryVal === 'number' ? entryVal.toLocaleString() : entryVal;
          return (
            <div key={index} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '12px', margin: '4px 0' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.9rem', color: '#cbd5e1' }}>
                <span style={{ width: '8px', height: '8px', borderRadius: '50%', background: entry.color || entry.fill || '#3b82f6', display: 'inline-block' }}></span>
                <span>{entry.name || yAxis || 'Metric'}:</span>
              </div>
              <div style={{ fontSize: '0.95rem', fontWeight: 700, color: '#ffffff' }}>
                {formattedVal && payload.length === 1 ? formattedVal : displayVal}
              </div>
            </div>
          );
        })}

        {pct !== undefined && (
          <div style={{ marginTop: '8px', paddingTop: '6px', borderTop: '1px solid rgba(255, 255, 255, 0.1)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: '#94a3b8', marginBottom: '3px' }}>
              <span>Share of Total</span>
              <span style={{ fontWeight: 600, color: '#38bdf8' }}>{pct}%</span>
            </div>
            <div style={{ width: '100%', height: '4px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '999px', overflow: 'hidden' }}>
              <div style={{ width: `${Math.min(100, pct)}%`, height: '100%', background: 'linear-gradient(90deg, #38bdf8, #818cf8)', borderRadius: '999px' }}></div>
            </div>
          </div>
        )}
      </div>
    );
  }
  return null;
};

const axisProps = {
  axisLine: false,
  tickLine: false,
  tick: { fill: 'var(--c-text-secondary)', fontSize: 11 }
};

// SVG Container for custom visual layouts
const SvgContainer = ({ children }) => (
  <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '220px' }}>
    <svg width="100%" height="100%" viewBox="0 0 340 220" style={{ overflow: 'visible' }}>
      {children}
    </svg>
  </div>
);

// Custom Treemap Content Item
const CustomTreemapContent = (props) => {
  const { x, y, width, height, index, name, value, percentage } = props;
  if (width < 35 || height < 25) return null;
  const color = COLORS[index % COLORS.length];

  return (
    <g>
      <rect
        x={x + 2}
        y={y + 2}
        width={width - 4}
        height={height - 4}
        rx={6}
        fill={color}
        fillOpacity={0.85}
        stroke="var(--c-glass-border)"
        strokeWidth={1}
        style={{ transition: 'all 0.2s ease', cursor: 'pointer' }}
      />
      {width > 60 && height > 35 && (
        <text
          x={x + width / 2}
          y={y + height / 2 - 4}
          textAnchor="middle"
          fill="#ffffff"
          fontSize={width > 100 ? 12 : 10}
          fontWeight="600"
        >
          {name}
        </text>
      )}
      {width > 60 && height > 50 && (
        <text
          x={x + width / 2}
          y={y + height / 2 + 12}
          textAnchor="middle"
          fill="rgba(255,255,255,0.85)"
          fontSize={10}
          fontWeight="500"
        >
          {typeof value === 'number' ? value.toLocaleString() : value}
        </text>
      )}
    </g>
  );
};

// Heatmap Renderer
const RealHeatmapRenderer = ({ data, matrixData, xAxis, yAxis }) => {
  const xLabels = matrixData?.x_labels || ['Cat A', 'Cat B', 'Cat C', 'Cat D'];
  const yLabels = matrixData?.y_labels || ['Group 1', 'Group 2', 'Group 3'];
  const items = data || matrixData?.data || [];

  return (
    <div style={{ width: '100%', height: '100%', display: 'flex', flexDirection: 'column', padding: '8px 12px' }}>
      <div style={{ overflowX: 'auto', flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <table style={{ borderCollapse: 'separate', borderSpacing: '6px', width: '100%', maxWidth: '480px' }}>
          <thead>
            <tr>
              <th style={{ padding: '6px', fontSize: '0.75rem', color: 'var(--c-text-secondary)', textAlign: 'left' }}></th>
              {xLabels.map((xl) => (
                <th key={xl} style={{ padding: '6px', fontSize: '0.75rem', color: 'var(--c-text-secondary)', fontWeight: 600, textAlign: 'center', whiteSpace: 'nowrap' }}>
                  {xl}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {yLabels.map((yl) => (
              <tr key={yl}>
                <td style={{ padding: '6px', fontSize: '0.75rem', fontWeight: 600, color: 'var(--c-text-secondary)', whiteSpace: 'nowrap' }}>
                  {yl}
                </td>
                {xLabels.map((xl) => {
                  const cell = items.find((it) => it.x === xl && it.y === yl) || {};
                  const intensity = cell.intensity !== undefined ? cell.intensity : 0.3;
                  const displayVal = cell.formatted_value || (cell.value ? cell.value.toLocaleString() : '-');

                  return (
                    <td
                      key={`${yl}-${xl}`}
                      title={`${yl} × ${xl}: ${displayVal}`}
                      style={{
                        padding: '10px 8px',
                        textAlign: 'center',
                        borderRadius: '8px',
                        background: `rgba(59, 130, 246, ${Math.max(0.15, intensity)})`,
                        color: intensity > 0.5 ? '#ffffff' : 'var(--c-text)',
                        fontSize: '0.8rem',
                        fontWeight: 600,
                        border: '1px solid var(--c-glass-border)',
                        cursor: 'pointer',
                        transition: 'transform 0.15s ease',
                      }}
                    >
                      {displayVal}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: '8px', fontSize: '0.75rem', color: 'var(--c-text-secondary)', marginTop: '4px' }}>
        <span>Low Intensity</span>
        <div style={{ width: '60px', height: '6px', borderRadius: '3px', background: 'linear-gradient(90deg, rgba(59, 130, 246, 0.15), rgba(59, 130, 246, 0.95))' }}></div>
        <span>High Intensity</span>
      </div>
    </div>
  );
};

// Box Plot Renderer
const RealBoxPlotRenderer = ({ data, xAxis, yAxis }) => {
  const items = Array.isArray(data) && data.length ? data : [
    { name: 'Category 1', min: 10, q1: 25, median: 45, q3: 65, max: 90 },
    { name: 'Category 2', min: 20, q1: 35, median: 55, q3: 75, max: 100 },
  ];

  const overallMax = Math.max(...items.map((it) => it.max || 100)) || 100;
  const overallMin = Math.min(...items.map((it) => it.min || 0)) || 0;
  const range = overallMax - overallMin || 1.0;

  return (
    <SvgContainer>
      {/* Axes */}
      <line x1="45" y1="20" x2="45" y2="170" stroke="var(--c-glass-border)" strokeWidth="1.5" />
      <line x1="45" y1="170" x2="310" y2="170" stroke="var(--c-glass-border)" strokeWidth="1.5" />

      {/* Y Axis ticks */}
      {[0, 0.5, 1].map((pct, i) => {
        const val = overallMin + range * (1 - pct);
        const y = 25 + pct * 140;
        return (
          <g key={i}>
            <line x1="40" y1={y} x2="45" y2={y} stroke="var(--c-text-secondary)" strokeWidth="1" />
            <text x="36" y={y + 4} textAnchor="end" fill="var(--c-text-secondary)" fontSize="9">
              {val >= 1000 ? `${(val / 1000).toFixed(1)}k` : Math.round(val)}
            </text>
          </g>
        );
      })}

      {/* Boxes */}
      {items.map((it, i) => {
        const xCenter = 70 + i * (230 / items.length);
        const boxWidth = Math.min(36, 180 / items.length);

        const yMin = 165 - ((it.min - overallMin) / range) * 140;
        const yQ1 = 165 - ((it.q1 - overallMin) / range) * 140;
        const yMed = 165 - ((it.median - overallMin) / range) * 140;
        const yQ3 = 165 - ((it.q3 - overallMin) / range) * 140;
        const yMax = 165 - ((it.max - overallMin) / range) * 140;

        const boxHeight = Math.max(4, yQ1 - yQ3);
        const color = COLORS[i % COLORS.length];

        return (
          <g key={i} className="box-plot-group" style={{ cursor: 'pointer' }}>
            <title>{`${it.name}: Min=${it.min}, Q1=${it.q1}, Median=${it.median}, Q3=${it.q3}, Max=${it.max}`}</title>
            {/* Whiskers */}
            <line x1={xCenter} y1={yMin} x2={xCenter} y2={yQ1} stroke="var(--c-text-secondary)" strokeWidth="1.5" strokeDasharray="3 3" />
            <line x1={xCenter} y1={yQ3} x2={xCenter} y2={yMax} stroke="var(--c-text-secondary)" strokeWidth="1.5" strokeDasharray="3 3" />
            <line x1={xCenter - 8} y1={yMin} x2={xCenter + 8} y2={yMin} stroke="var(--c-text-secondary)" strokeWidth="1.5" />
            <line x1={xCenter - 8} y1={yMax} x2={xCenter + 8} y2={yMax} stroke="var(--c-text-secondary)" strokeWidth="1.5" />

            {/* Interquartile Box */}
            <rect
              x={xCenter - boxWidth / 2}
              y={yQ3}
              width={boxWidth}
              height={boxHeight}
              rx={4}
              fill={color}
              fillOpacity={0.8}
              stroke="var(--c-glass-border)"
              strokeWidth={1}
            />

            {/* Median Line */}
            <line
              x1={xCenter - boxWidth / 2}
              y1={yMed}
              x2={xCenter + boxWidth / 2}
              y2={yMed}
              stroke="#ffffff"
              strokeWidth={2.5}
            />

            {/* Category label */}
            <text x={xCenter} y="188" textAnchor="middle" fill="var(--c-text-secondary)" fontSize="10" fontWeight="500">
              {it.name.length > 8 ? `${it.name.slice(0, 7)}…` : it.name}
            </text>
          </g>
        );
      })}
    </SvgContainer>
  );
};

// Master Chart Renderer Component
export const ChartRenderer = ({ type, data, xAxis, yAxis, secondaryDimension, matrixData }) => {
  const metricName = yAxis || 'Value';
  const hasRealData = Array.isArray(data) && data.length > 0;
  const chartData = hasRealData ? data : timeSeriesData;
  const categoryData = hasRealData ? data : categoricalData;
  const scatterPoints = hasRealData ? data : scatterData;

  switch (type) {
    // 1. AREA GRAPH (Smooth Volume Gradient)
    case 'Area Graph':
    case 'Area':
    case 'Stacked Area Graph':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 12, right: 12, left: -16, bottom: 0 }}>
            <defs>
              <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.45} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" opacity={0.6} />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<PowerBiTooltip yAxis={metricName} xAxis={xAxis} />} />
            <Area
              type="monotone"
              name={metricName}
              dataKey="value"
              stroke="#3b82f6"
              strokeWidth={3}
              fill="url(#areaGrad)"
              activeDot={{ r: 6, stroke: '#ffffff', strokeWidth: 2, fill: '#3b82f6' }}
            />
          </AreaChart>
        </ResponsiveContainer>
      );

    // 2. LINE GRAPH (Trend & Points)
    case 'Line Graph':
    case 'Line':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 12, right: 12, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" opacity={0.6} />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<PowerBiTooltip yAxis={metricName} xAxis={xAxis} />} />
            <Line
              type="monotone"
              name={metricName}
              dataKey="value"
              stroke="#10b981"
              strokeWidth={3}
              dot={{ r: 4, fill: '#10b981', stroke: 'var(--c-bg)', strokeWidth: 1.5 }}
              activeDot={{ r: 7, fill: '#10b981', stroke: '#ffffff', strokeWidth: 2 }}
            />
          </LineChart>
        </ResponsiveContainer>
      );

    // 3. BAR CHART (Categorical Ranking)
    case 'Bar Chart':
    case 'Bar':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={categoryData} margin={{ top: 12, right: 12, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" opacity={0.6} />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<PowerBiTooltip yAxis={metricName} xAxis={xAxis} />} />
            <Bar dataKey="value" name={metricName} radius={[6, 6, 0, 0]}>
              {categoryData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      );

    // 4. STACKED BAR GRAPH & MULTI-SET BAR
    case 'Stacked Bar Graph':
    case 'Multi-set Bar Chart': {
      const secondaryKeys = (data && data[0]?.secondary_keys) || ['Direct', 'Partner'];
      const isStacked = type === 'Stacked Bar Graph';

      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 12, right: 12, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" opacity={0.6} />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<PowerBiTooltip yAxis={metricName} xAxis={xAxis} />} />
            <Legend verticalAlign="top" height={32} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            {secondaryKeys.map((secKey, i) => (
              <Bar
                key={secKey}
                dataKey={secKey}
                name={secKey}
                stackId={isStacked ? 'a' : undefined}
                fill={COLORS[i % COLORS.length]}
                radius={isStacked ? (i === secondaryKeys.length - 1 ? [4, 4, 0, 0] : [0, 0, 0, 0]) : [4, 4, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      );
    }

    // 5. DONUT & PIE CHART (Market Share / Proportions)
    case 'Donut Chart':
    case 'Donut':
    case 'Pie Chart':
    case 'Pie': {
      const isDonut = type.toLowerCase().includes('donut');
      return (
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={categoryData}
              cx="50%"
              cy="50%"
              innerRadius={isDonut ? 55 : 0}
              outerRadius={80}
              paddingAngle={isDonut ? 4 : 2}
              dataKey="value"
              nameKey="name"
              stroke="var(--c-bg)"
              strokeWidth={2}
            >
              {categoryData.map((_, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <RechartsTooltip content={<PowerBiTooltip yAxis={metricName} xAxis={xAxis} />} />
            <Legend verticalAlign="bottom" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
          </PieChart>
        </ResponsiveContainer>
      );
    }

    // 6. SCATTERPLOT (Bivariate Correlation)
    case 'Scatterplot':
    case 'Scatter':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 12, bottom: 12, left: -16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--c-glass-border)" opacity={0.6} />
            <XAxis type="number" dataKey="x" name={xAxis || 'X'} {...axisProps} />
            <YAxis type="number" dataKey="y" name={yAxis || 'Y'} {...axisProps} />
            <ZAxis type="number" dataKey="z" range={[60, 200]} />
            <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} content={<PowerBiTooltip yAxis={yAxis} xAxis={xAxis} />} />
            <Scatter name={`${yAxis || 'Y'} vs ${xAxis || 'X'}`} data={scatterPoints} fill="#8b5cf6" opacity={0.8} />
          </ScatterChart>
        </ResponsiveContainer>
      );

    // 7. HISTOGRAM (Frequency Distribution)
    case 'Histogram':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={categoryData} margin={{ top: 12, right: 12, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" opacity={0.6} />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<PowerBiTooltip yAxis="Frequency Count" xAxis="Range" />} />
            <Bar dataKey="value" name="Frequency" fill="#06b6d4" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      );

    // 8. TREEMAP (Hierarchical Breakdown)
    case 'Treemap':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <Treemap
            data={categoryData}
            dataKey="value"
            aspectRatio={4 / 3}
            stroke="var(--c-bg)"
            content={<CustomTreemapContent />}
          >
            <RechartsTooltip content={<PowerBiTooltip yAxis={metricName} xAxis={xAxis} />} />
          </Treemap>
        </ResponsiveContainer>
      );

    // 9. RADAR CHART (Multi-Dimensional Scoring)
    case 'Radar Chart':
    case 'Radar':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="75%" data={categoryData}>
            <PolarGrid stroke="var(--c-glass-border)" />
            <PolarAngleAxis dataKey="subject" tick={{ fill: 'var(--c-text-secondary)', fontSize: 11 }} />
            <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
            <RechartsTooltip content={<PowerBiTooltip yAxis="Relative Score" xAxis="Dimension" />} />
            <Radar name="Performance" dataKey="A" stroke="#8b5cf6" fill="#8b5cf6" fillOpacity={0.45} />
          </RadarChart>
        </ResponsiveContainer>
      );

    // 10. HEATMAP (Matrix Density)
    case 'Heatmap':
      return <RealHeatmapRenderer data={data} matrixData={matrixData} xAxis={xAxis} yAxis={yAxis} />;

    // 11. BOX PLOT (Distribution Quartiles)
    case 'Box Plot':
      return <RealBoxPlotRenderer data={data} xAxis={xAxis} yAxis={yAxis} />;

    // 12. FUNNEL CHART
    case 'Funnel Chart':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <FunnelChart>
            <RechartsTooltip content={<PowerBiTooltip yAxis={metricName} xAxis={xAxis} />} />
            <Funnel dataKey="value" data={categoryData} isAnimationActive>
              {categoryData.map((_, index) => (
                <Cell key={`funnel-cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Funnel>
          </FunnelChart>
        </ResponsiveContainer>
      );

    // Default Fallback to Bar Chart
    default:
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={categoryData} margin={{ top: 12, right: 12, left: -16, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" opacity={0.6} />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<PowerBiTooltip yAxis={metricName} xAxis={xAxis} />} />
            <Bar dataKey="value" fill="#3b82f6" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      );
  }
};
