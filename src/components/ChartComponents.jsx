import React from 'react';
import { 
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer,
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, ScatterChart, Scatter, ZAxis,
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Treemap, FunnelChart, Funnel,
  RadialBarChart, RadialBar, Legend
} from 'recharts';
import { timeSeriesData, categoricalData, scatterData, radarData, treemapData, funnelData } from '../data/mockData';

const COLORS = ['var(--c-accent-blue)', 'var(--c-accent-purple)', 'var(--c-accent-emerald)', 'var(--c-accent-cyan)', 'var(--c-accent-orange)'];

export const CATEGORIZED_CHARTS = {
  "Method": [
    "Area Graph", "Bar Chart", "Box and Whisker Plot", "Bubble Chart", "Bullet Graph", 
    "Candlestick Chart", "Density Plot", "Error Bars", "Histogram", "Kagi Chart", 
    "Line Graph", "Marimekko Chart", "Multi-set Bar Chart", "OHLC Chart", 
    "Parallel Coordinates Plot", "Point & Figure Chart", "Population Pyramid", 
    "Radar Chart", "Radial Bar Chart", "Radial Column Chart", "Scatterplot", 
    "Span Chart", "Spiral Plot", "Stacked Area Graph", "Stacked Bar Graph", 
    "Stream Graph", "Violin Plot"
  ],
  "Diagrams": [
    "Arc Diagram", "Brainstorm", "Chord Diagram", "Flow Chart", "Illustration Diagram", 
    "Network Diagram", "Non-ribbon Chord Diagram", "Sankey Diagram", "Timeline", 
    "Tree Diagram", "Venn Diagram"
  ],
  "Tables": [
    "Calendar", "Gantt Chart", "Heatmap", "Stem & Leaf Plot", "Tally Chart", "Time Table"
  ],
  "Other": [
    "Circle Packing", "Donut Chart", "Dot Matrix Chart", "Nightingale Rose Chart", 
    "Parallel Sets", "Pictogram Chart", "Pie Chart", "Proportional Area Chart", 
    "Sunburst Diagram", "Treemap", "Word Cloud"
  ],
  "Maps/Geographical": [
    "Bubble Map", "Choropleth Map", "Connection Map", "Dot Map", "Flow Map"
  ]
};

export const AVAILABLE_CHARTS = Object.values(CATEGORIZED_CHARTS).flat();

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="apple-tooltip glass-panel" style={{ zIndex: 100 }}>
        <p className="tooltip-label">{label || payload[0].name}</p>
        {payload.map((entry, index) => (
          <p key={index} className="tooltip-value" style={{ fontSize: '0.95rem' }}>
            <span style={{color: entry.color || entry.fill}}>●</span> {entry.name || 'Value'}: {entry.value ? entry.value.toLocaleString() : entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

const axisProps = {
  axisLine: false, tickLine: false, tick: { fill: 'var(--c-text-secondary)', fontSize: 12 }
};

// SVG Container with ViewBox
const SvgContainer = ({ children }) => (
  <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '180px' }}>
    <svg width="100%" height="100%" viewBox="0 0 320 200" style={{ overflow: 'visible' }}>
      {children}
    </svg>
  </div>
);

// Standard Legend Component for SVG Visualizations
const SvgLegend = ({ items = [
  { color: 'var(--c-accent-blue)', label: 'Actual' },
  { color: 'var(--c-accent-purple)', label: 'Target' }
] }) => (
  <g className="svg-legend">
    {items.map((item, i) => (
      <g key={i} transform={`translate(${45 + i * 85}, 10)`}>
        <rect x="0" y="0" width="8" height="8" rx="2" fill={item.color} />
        <text x="12" y="7.5" fill="var(--c-text-secondary)" fontSize="9" fontWeight="500">{item.label}</text>
      </g>
    ))}
  </g>
);

// Standard Axes Component for SVG Charts
const SvgAxes = ({ xLabels = ['A', 'B', 'C', 'D'], yTicks = ['0', '50', '100'] }) => (
  <g className="svg-axes">
    {/* Y Axis Line */}
    <line x1="45" y1="25" x2="45" y2="160" stroke="var(--c-glass-border)" strokeWidth="1.5" />
    {/* X Axis Line */}
    <line x1="45" y1="160" x2="300" y2="160" stroke="var(--c-glass-border)" strokeWidth="1.5" />
    {/* Y Axis Labels */}
    {yTicks.map((tick, i) => {
      const y = 160 - (i * (135 / (yTicks.length - 1)));
      return (
        <g key={i}>
          <line x1="40" y1={y} x2="45" y2={y} stroke="var(--c-text-secondary)" strokeWidth="1" />
          <text x="35" y={y + 4} textAnchor="end" fill="var(--c-text-secondary)" fontSize="10">{tick}</text>
        </g>
      );
    })}
    {/* X Axis Labels */}
    {xLabels.map((label, i) => {
      const x = 45 + ((i + 0.5) * (255 / xLabels.length));
      return (
        <text key={i} x={x} y="176" textAnchor="middle" fill="var(--c-text-secondary)" fontSize="10">{label}</text>
      );
    })}
  </g>
);

export const ChartRenderer = ({ type, data, xAxis, yAxis }) => {
  // Dashboard charts receive verified values from the backend.  Mock data is
  // retained only as a visual fallback for manually added empty charts.
  const chartData = Array.isArray(data) && data.length ? data : timeSeriesData;
  const categoryChartData = Array.isArray(data) && data.length ? data : categoricalData;
  const pointData = Array.isArray(data) && data.length ? data : scatterData;
  const metricName = yAxis || 'Value';
  switch (type) {
    // === METHOD CHARTS ===
    case 'Area Graph':
    case 'Area':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorVal" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="var(--c-accent-blue)" stopOpacity={0.3}/>
                <stop offset="95%" stopColor="var(--c-accent-blue)" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <Area type="monotone" name={metricName} dataKey="value" stroke="var(--c-accent-blue)" strokeWidth={3} fillOpacity={1} fill="url(#colorVal)" />
          </AreaChart>
        </ResponsiveContainer>
      );
      
    case 'Stacked Area Graph':
    case 'Stream Graph':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={timeSeriesData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <Area type="monotone" name="Direct Sales" dataKey="value" stackId="1" stroke="var(--c-accent-blue)" fill="var(--c-accent-blue)" fillOpacity={0.7} />
            <Area type="monotone" name="Partner Sales" dataKey="secondary" stackId="1" stroke="var(--c-accent-purple)" fill="var(--c-accent-purple)" fillOpacity={0.7} />
          </AreaChart>
        </ResponsiveContainer>
      );
    
    case 'Line Graph':
    case 'Line':
    case 'Span Chart':
    case 'Kagi Chart':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <Line type="monotone" name={metricName} dataKey="value" stroke="var(--c-accent-emerald)" strokeWidth={3} dot={{r: 4}} activeDot={{r: 6}} />
          </LineChart>
        </ResponsiveContainer>
      );

    case 'Bar Chart':
    case 'Bar':
    case 'Histogram':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <Bar dataKey="value" name={metricName} fill="var(--c-accent-cyan)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      );
      
    case 'Multi-set Bar Chart':
    case 'Marimekko Chart':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={timeSeriesData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <Bar dataKey="value" name="Online" fill="var(--c-accent-cyan)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="secondary" name="In-Store" fill="var(--c-accent-purple)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      );

    case 'Stacked Bar Graph':
    case 'Population Pyramid':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={timeSeriesData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--c-glass-border)" />
            <XAxis dataKey="name" {...axisProps} />
            <YAxis {...axisProps} />
            <RechartsTooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <Bar dataKey="value" name="Segment A" stackId="a" fill="var(--c-accent-blue)" />
            <Bar dataKey="secondary" name="Segment B" stackId="a" fill="var(--c-accent-emerald)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      );

    case 'Box and Whisker Plot':
    case 'Violin Plot':
    case 'Density Plot':
    case 'Error Bars':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'Interquartile' },
            { color: 'var(--c-accent-orange)', label: 'Outlier' }
          ]} />
          <SvgAxes xLabels={['Grp A', 'Grp B', 'Grp C', 'Grp D']} yTicks={['0', '50', '100']} />
          {[75, 138, 201, 264].map((x, i) => (
            <g key={i}>
              <line x1={x} y1="35" x2={x} y2="145" stroke="var(--c-text-secondary)" strokeWidth="1.5" strokeDasharray="3 3" />
              <rect x={x - 14} y="60" width="28" height="50" rx="5" fill={COLORS[i % COLORS.length]} opacity="0.85" stroke="var(--c-glass-border)" />
              <line x1={x - 14} y1="85" x2={x + 14} y2="85" stroke="var(--c-white)" strokeWidth="2.5" />
              <circle cx={x} cy="25" r="3.5" fill="var(--c-accent-orange)" />
            </g>
          ))}
        </SvgContainer>
      );

    case 'Candlestick Chart':
    case 'OHLC Chart':
    case 'Point & Figure Chart':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-emerald)', label: 'Bullish (Gain)' },
            { color: 'var(--c-accent-orange)', label: 'Bearish (Loss)' }
          ]} />
          <SvgAxes xLabels={['Mon', 'Tue', 'Wed', 'Thu', 'Fri']} yTicks={['$10', '$25', '$40']} />
          {[70, 120, 170, 220, 270].map((x, i) => {
            const isGreen = i % 2 === 0;
            const yTop = 35 + i * 8;
            const yBottom = 135 - i * 4;
            const bodyTop = yTop + 18;
            const bodyHeight = 35;
            const color = isGreen ? 'var(--c-accent-emerald)' : 'var(--c-accent-orange)';
            return (
              <g key={i}>
                <line x1={x} y1={yTop} x2={x} y2={yBottom} stroke={color} strokeWidth="1.5" />
                <rect x={x - 8} y={bodyTop} width="16" height={bodyHeight} rx="3" fill={color} />
              </g>
            );
          })}
        </SvgContainer>
      );

    case 'Parallel Coordinates Plot':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'Group 1' },
            { color: 'var(--c-accent-purple)', label: 'Group 2' },
            { color: 'var(--c-accent-emerald)', label: 'Group 3' }
          ]} />
          {['Var 1', 'Var 2', 'Var 3', 'Var 4', 'Var 5'].map((name, i) => {
            const x = 50 + i * 55;
            return (
              <g key={i}>
                <text x={x} y="30" textAnchor="middle" fill="var(--c-text-secondary)" fontSize="10" fontWeight="bold">{name}</text>
                <line x1={x} y1="38" x2={x} y2="155" stroke="var(--c-glass-border)" strokeWidth="1.5" />
              </g>
            );
          })}
          <path d="M50 45 L105 90 L160 40 L215 120 L270 70" fill="none" stroke="var(--c-accent-blue)" strokeWidth="2.5" opacity="0.85" />
          <path d="M50 100 L105 45 L160 110 L215 60 L270 140" fill="none" stroke="var(--c-accent-purple)" strokeWidth="2.5" opacity="0.85" />
          <path d="M50 130 L105 120 L160 80 L215 140 L270 45" fill="none" stroke="var(--c-accent-emerald)" strokeWidth="2.5" opacity="0.85" />
        </SvgContainer>
      );

    case 'Bullet Graph':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'Actual' },
            { color: 'var(--c-text)', label: 'Target Line' }
          ]} />
          <line x1="60" y1="160" x2="300" y2="160" stroke="var(--c-glass-border)" strokeWidth="1.5" />
          {['0%', '25%', '50%', '75%', '100%'].map((t, i) => (
            <g key={i}>
              <line x1={60 + i * 60} y1="160" x2={60 + i * 60} y2="165" stroke="var(--c-text-secondary)" strokeWidth="1" />
              <text x={60 + i * 60} y="178" textAnchor="middle" fill="var(--c-text-secondary)" fontSize="9">{t}</text>
            </g>
          ))}
          {['Revenue', 'Profit', 'Users'].map((label, i) => {
            const y = 30 + i * 42;
            return (
              <g key={i}>
                <text x="50" y={y + 16} textAnchor="end" fill="var(--c-text-secondary)" fontSize="10" fontWeight="500">{label}</text>
                <rect x="60" y={y} width="240" height="22" rx="4" fill="var(--c-glass-border)" opacity="0.3" />
                <rect x="60" y={y} width={170 - i * 35} height="22" rx="4" fill={COLORS[i]} opacity="0.4" />
                <rect x="60" y={y + 5} width={130 - i * 25} height="12" rx="2" fill={COLORS[i]} />
                <line x1={180 - i * 25} y1={y - 3} x2={180 - i * 25} y2={y + 25} stroke="var(--c-text)" strokeWidth="2.5" />
              </g>
            );
          })}
        </SvgContainer>
      );

    case 'Radar Chart':
    case 'Radar':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
            <PolarGrid stroke="var(--c-glass-border)" />
            <PolarAngleAxis dataKey="subject" tick={{fill: 'var(--c-text-secondary)', fontSize: 11}} />
            <PolarRadiusAxis angle={30} domain={[0, 150]} tick={false} axisLine={false} />
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <Radar name="Performance Score" dataKey="A" stroke="var(--c-accent-blue)" fill="var(--c-accent-blue)" fillOpacity={0.5} />
            <RechartsTooltip content={<CustomTooltip />} />
          </RadarChart>
        </ResponsiveContainer>
      );

    case 'Radial Bar Chart':
    case 'Radial Column Chart':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <RadialBarChart cx="50%" cy="50%" innerRadius="10%" outerRadius="80%" barSize={10} data={categoricalData}>
            <RadialBar minAngle={15} background clockWise dataKey="value" />
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <RechartsTooltip content={<CustomTooltip />} />
          </RadialBarChart>
        </ResponsiveContainer>
      );

    case 'Scatterplot':
    case 'Scatter':
    case 'Bubble Chart':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: -20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--c-glass-border)" />
            <XAxis type="number" dataKey="x" name="stature" unit="cm" {...axisProps} />
            <YAxis type="number" dataKey="y" name="weight" unit="kg" {...axisProps} />
            <ZAxis type="number" dataKey="z" range={[60, 400]} name="score" />
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <RechartsTooltip cursor={{ strokeDasharray: '3 3' }} content={<CustomTooltip />} />
            <Scatter name={`${yAxis || 'Y'} vs ${xAxis || 'X'}`} data={pointData} fill="var(--c-accent-purple)" />
          </ScatterChart>
        </ResponsiveContainer>
      );

    case 'Spiral Plot':
      return (
        <SvgContainer>
          <SvgLegend items={[{ color: 'var(--c-accent-blue)', label: 'Spiral Trajectory' }]} />
          <line x1="150" y1="20" x2="150" y2="160" stroke="var(--c-glass-border)" strokeWidth="1" strokeDasharray="2 2" />
          <line x1="80" y1="90" x2="220" y2="90" stroke="var(--c-glass-border)" strokeWidth="1" strokeDasharray="2 2" />
          <text x="150" y="175" textAnchor="middle" fill="var(--c-text-secondary)" fontSize="9">Angle θ (Rad)</text>
          <path d="M150 90 Q150 70 130 70 Q100 70 100 100 Q100 140 150 140 Q210 140 210 90 Q210 30 140 30 Q50 30 50 110" fill="none" stroke="var(--c-accent-blue)" strokeWidth="3.5" strokeDasharray="6 6" />
          <circle cx="150" cy="90" r="5" fill="var(--c-accent-purple)" />
          <circle cx="130" cy="70" r="7" fill="var(--c-accent-cyan)" />
          <circle cx="100" cy="100" r="8" fill="var(--c-accent-emerald)" />
          <circle cx="150" cy="140" r="10" fill="var(--c-accent-orange)" />
        </SvgContainer>
      );

    // === DIAGRAMS ===
    case 'Arc Diagram':
      return (
        <SvgContainer>
          <SvgLegend items={[{ color: 'var(--c-accent-blue)', label: 'Node Links' }]} />
          <line x1="30" y1="140" x2="290" y2="140" stroke="var(--c-glass-border)" strokeWidth="1.5" />
          {['Node 1', 'Node 2', 'Node 3', 'Node 4', 'Node 5'].map((name, i) => {
            const x = 50 + i * 55;
            return (
              <g key={i}>
                <circle cx={x} cy="140" r="7" fill={COLORS[i % COLORS.length]} />
                <text x={x} y="158" textAnchor="middle" fill="var(--c-text-secondary)" fontSize="9">{name}</text>
              </g>
            );
          })}
          <path d="M50 140 A 27.5 27.5 0 0 1 105 140" fill="none" stroke="var(--c-accent-blue)" strokeWidth="2.5" />
          <path d="M105 140 A 55 55 0 0 1 215 140" fill="none" stroke="var(--c-accent-purple)" strokeWidth="2.5" />
          <path d="M50 140 A 82.5 82.5 0 0 1 215 140" fill="none" stroke="var(--c-accent-cyan)" strokeWidth="2" strokeDasharray="4 4" />
          <path d="M160 140 A 55 55 0 0 1 270 140" fill="none" stroke="var(--c-accent-emerald)" strokeWidth="2.5" />
        </SvgContainer>
      );

    case 'Brainstorm':
    case 'Tree Diagram':
    case 'Network Diagram':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'Parent Node' },
            { color: 'var(--c-accent-purple)', label: 'Child Nodes' }
          ]} />
          <line x1="160" y1="90" x2="70" y2="40" stroke="var(--c-glass-border)" strokeWidth="1.5" />
          <line x1="160" y1="90" x2="250" y2="40" stroke="var(--c-glass-border)" strokeWidth="1.5" />
          <line x1="160" y1="90" x2="70" y2="140" stroke="var(--c-glass-border)" strokeWidth="1.5" />
          <line x1="160" y1="90" x2="250" y2="140" stroke="var(--c-glass-border)" strokeWidth="1.5" />
          <circle cx="160" cy="90" r="18" fill="var(--c-accent-blue)" />
          <text x="160" y="93" textAnchor="middle" fill="#fff" fontSize="10" fontWeight="bold">Root</text>
          
          <circle cx="70" cy="40" r="13" fill="var(--c-accent-purple)" />
          <text x="70" y="43" textAnchor="middle" fill="#fff" fontSize="9">N1</text>
          
          <circle cx="250" cy="40" r="13" fill="var(--c-accent-cyan)" />
          <text x="250" y="43" textAnchor="middle" fill="#fff" fontSize="9">N2</text>
          
          <circle cx="70" cy="140" r="13" fill="var(--c-accent-emerald)" />
          <text x="70" y="143" textAnchor="middle" fill="#fff" fontSize="9">N3</text>
          
          <circle cx="250" cy="140" r="13" fill="var(--c-accent-orange)" />
          <text x="250" y="143" textAnchor="middle" fill="#fff" fontSize="9">N4</text>
        </SvgContainer>
      );

    case 'Chord Diagram':
    case 'Non-ribbon Chord Diagram':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'Inter-relationship' }
          ]} />
          <circle cx="160" cy="95" r="60" fill="none" stroke="var(--c-glass-border)" strokeWidth="3" />
          <path d="M160 35 Q130 80 100 95" fill="none" stroke="var(--c-accent-blue)" strokeWidth="2.5" opacity="0.85" />
          <path d="M160 35 Q190 105 220 95" fill="none" stroke="var(--c-accent-purple)" strokeWidth="2.5" opacity="0.85" />
          <path d="M100 95 Q160 135 160 155" fill="none" stroke="var(--c-accent-emerald)" strokeWidth="2.5" opacity="0.85" />
          <path d="M220 95 Q130 120 160 155" fill="none" stroke="var(--c-accent-cyan)" strokeWidth="2.5" opacity="0.85" />
        </SvgContainer>
      );

    case 'Flow Chart':
    case 'Illustration Diagram':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'Action' },
            { color: 'var(--c-accent-purple)', label: 'Decision' }
          ]} />
          <rect x="20" y="75" width="60" height="40" rx="6" fill="var(--c-accent-blue)" />
          <text x="50" y="99" textAnchor="middle" fill="#fff" fontSize="10" fontWeight="500">Start</text>

          <path d="M80 95 L115 95" stroke="var(--c-text-secondary)" strokeWidth="1.5" />
          
          <polygon points="150,70 180,95 150,120 120,95" fill="var(--c-accent-purple)" />
          <text x="150" y="98" textAnchor="middle" fill="#fff" fontSize="9" fontWeight="500">Check</text>

          <path d="M180 95 L215 95" stroke="var(--c-text-secondary)" strokeWidth="1.5" />

          <rect x="215" y="75" width="60" height="40" rx="20" fill="var(--c-accent-emerald)" />
          <text x="245" y="99" textAnchor="middle" fill="#fff" fontSize="10" fontWeight="500">End</text>
        </SvgContainer>
      );

    case 'Sankey Diagram':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <FunnelChart>
            <RechartsTooltip content={<CustomTooltip />} />
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <Funnel dataKey="value" data={funnelData} isAnimationActive />
          </FunnelChart>
        </ResponsiveContainer>
      );

    case 'Timeline':
      return (
        <SvgContainer>
          <SvgLegend items={[{ color: 'var(--c-accent-blue)', label: 'Milestone Events' }]} />
          <line x1="30" y1="95" x2="290" y2="95" stroke="var(--c-glass-border)" strokeWidth="3" />
          {['2021', '2022', '2023', '2024'].map((year, i) => {
            const x = 50 + i * 70;
            return (
              <g key={i}>
                <circle cx={x} cy="95" r="7" fill={COLORS[i % COLORS.length]} />
                <line x1={x} y1="95" x2={x} y2={i % 2 === 0 ? 60 : 130} stroke={COLORS[i % COLORS.length]} strokeWidth="1.5" />
                <rect x={x - 24} y={i % 2 === 0 ? 35 : 135} width="48" height="20" rx="4" fill="var(--c-glass-bg)" stroke="var(--c-glass-border)" />
                <text x={x} y={i % 2 === 0 ? 48 : 148} textAnchor="middle" fill="var(--c-text)" fontSize="9" fontWeight="bold">{year}</text>
              </g>
            );
          })}
        </SvgContainer>
      );

    case 'Venn Diagram':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'Set A' },
            { color: 'var(--c-accent-purple)', label: 'Set B' }
          ]} />
          <circle cx="130" cy="95" r="42" fill="var(--c-accent-blue)" opacity="0.55" />
          <text x="110" y="90" fill="#fff" fontSize="11" fontWeight="bold">Set A</text>

          <circle cx="190" cy="95" r="42" fill="var(--c-accent-purple)" opacity="0.55" />
          <text x="210" y="90" fill="#fff" fontSize="11" fontWeight="bold">Set B</text>

          <circle cx="160" cy="125" r="35" fill="var(--c-accent-emerald)" opacity="0.45" />
          <text x="160" y="145" textAnchor="middle" fill="#fff" fontSize="10" fontWeight="bold">Set C</text>
        </SvgContainer>
      );

    // === TABLES ===
    case 'Calendar':
    case 'Time Table':
      return (
        <SvgContainer>
          <SvgLegend items={[{ color: 'var(--c-accent-blue)', label: 'Scheduled Activity' }]} />
          <rect x="25" y="25" width="270" height="145" rx="8" fill="var(--c-glass-bg)" stroke="var(--c-glass-border)" />
          {['S', 'M', 'T', 'W', 'T', 'F', 'S'].map((day, i) => (
            <text key={i} x={42 + i * 36} y={40} textAnchor="middle" fill="var(--c-text-secondary)" fontSize="10" fontWeight="bold">{day}</text>
          ))}
          {[0, 1, 2, 3, 4, 5, 6].map((col) => (
            [0, 1, 2, 3].map((row) => (
              <rect key={`${col}-${row}`} x={28 + col * 36} y={48 + row * 26} width="28" height="20" rx="3" fill={(col + row) % 3 === 0 ? 'var(--c-accent-blue)' : 'var(--c-glass-border)'} opacity={(col + row) % 3 === 0 ? 0.75 : 0.25} />
            ))
          ))}
        </SvgContainer>
      );

    case 'Gantt Chart':
      return (
        <SvgContainer>
          <SvgLegend items={[{ color: 'var(--c-accent-blue)', label: 'Task Duration' }]} />
          <line x1="70" y1="30" x2="300" y2="30" stroke="var(--c-glass-border)" strokeWidth="1" />
          {['Jan', 'Feb', 'Mar', 'Apr', 'May'].map((m, i) => (
            <text key={i} x={75 + i * 50} y="24" textAnchor="middle" fill="var(--c-text-secondary)" fontSize="9">{m}</text>
          ))}
          {['Task A', 'Task B', 'Task C', 'Task D'].map((task, i) => {
            const y = 42 + i * 30;
            return (
              <g key={i}>
                <text x="60" y={y + 12} textAnchor="end" fill="var(--c-text-secondary)" fontSize="10" fontWeight="500">{task}</text>
                <line x1="65" y1={y + 20} x2="300" y2={y + 20} stroke="var(--c-glass-border)" strokeWidth="1" strokeDasharray="2 2" />
              </g>
            );
          })}
          <rect x="75" y="44" width="75" height="14" rx="4" fill="var(--c-accent-blue)" />
          <rect x="125" y="74" width="105" height="14" rx="4" fill="var(--c-accent-purple)" />
          <rect x="180" y="104" width="85" height="14" rx="4" fill="var(--c-accent-emerald)" />
          <rect x="75" y="134" width="50" height="14" rx="4" fill="var(--c-accent-cyan)" />
        </SvgContainer>
      );

    case 'Heatmap':
      return (
        <SvgContainer>
          <SvgLegend items={[{ color: 'var(--c-accent-blue)', label: 'Intensity (Low -> High)' }]} />
          {['R1', 'R2', 'R3', 'R4'].map((row, i) => (
            <text key={i} x="35" y={44 + i * 30} textAnchor="end" fill="var(--c-text-secondary)" fontSize="9">{row}</text>
          ))}
          {['C1', 'C2', 'C3', 'C4', 'C5'].map((col, i) => (
            <text key={i} x={65 + i * 46} y="165" textAnchor="middle" fill="var(--c-text-secondary)" fontSize="9">{col}</text>
          ))}
          {[0, 1, 2, 3, 4].map((x) => (
            [0, 1, 2, 3].map((y) => (
              <rect key={`${x}-${y}`} x={45 + x * 46} y={28 + y * 30} width="40" height="24" rx="4" fill="var(--c-accent-blue)" opacity={((x + 1) * (y + 1)) / 20} />
            ))
          ))}
        </SvgContainer>
      );

    case 'Stem & Leaf Plot':
    case 'Tally Chart':
      return (
        <SvgContainer>
          <SvgLegend items={[{ color: 'var(--c-accent-blue)', label: 'Leaf Values' }]} />
          <text x="50" y="32" textAnchor="end" fill="var(--c-text)" fontSize="11" fontWeight="bold">Stem</text>
          <text x="95" y="32" fill="var(--c-text)" fontSize="11" fontWeight="bold">Leaf Data</text>
          <line x1="65" y1="22" x2="65" y2="165" stroke="var(--c-glass-border)" strokeWidth="2" />
          {[50, 80, 110, 140].map((y, i) => (
            <g key={i}>
              <text x="50" y={y + 5} textAnchor="end" fill="var(--c-text)" fontSize="13" fontWeight="bold">0{i + 1}</text>
              <text x="95" y={y + 5} fill="var(--c-accent-blue)" fontSize="13" letterSpacing="6">1 3 5 8 9</text>
            </g>
          ))}
        </SvgContainer>
      );

    // === OTHER ===
    case 'Circle Packing':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'Group A' },
            { color: 'var(--c-accent-purple)', label: 'Group B' }
          ]} />
          <circle cx="160" cy="95" r="68" fill="var(--c-glass-border)" opacity="0.3" />
          <circle cx="130" cy="78" r="32" fill="var(--c-accent-blue)" opacity="0.85" />
          <circle cx="190" cy="80" r="24" fill="var(--c-accent-purple)" opacity="0.85" />
          <circle cx="150" cy="128" r="26" fill="var(--c-accent-emerald)" opacity="0.85" />
          <circle cx="120" cy="78" r="9" fill="var(--c-accent-cyan)" />
        </SvgContainer>
      );

    case 'Dot Matrix Chart':
    case 'Pictogram Chart':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'Completed (64%)' },
            { color: 'var(--c-glass-border)', label: 'Remaining' }
          ]} />
          <SvgAxes xLabels={['0', '20', '40', '60', '80', '100']} yTicks={['Low', 'High']} />
          {[...Array(50)].map((_, i) => (
            <circle key={i} cx={60 + (i % 10) * 23} cy={35 + Math.floor(i / 10) * 24} r="7" fill={i < 32 ? 'var(--c-accent-blue)' : 'var(--c-glass-border)'} opacity={i < 32 ? 0.9 : 0.3} />
          ))}
        </SvgContainer>
      );

    case 'Nightingale Rose Chart':
    case 'Sunburst Diagram':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'North' },
            { color: 'var(--c-accent-purple)', label: 'South' },
            { color: 'var(--c-accent-emerald)', label: 'East' }
          ]} />
          <circle cx="160" cy="95" r="60" fill="none" stroke="var(--c-glass-border)" strokeWidth="1.5" />
          <path d="M160 95 L160 35 A 60 60 0 0 1 210 58 Z" fill="var(--c-accent-blue)" opacity="0.85" />
          <path d="M160 95 L210 58 A 60 60 0 0 1 220 113 Z" fill="var(--c-accent-purple)" opacity="0.85" />
          <path d="M160 95 L220 113 A 60 60 0 0 1 160 155 Z" fill="var(--c-accent-emerald)" opacity="0.85" />
          <path d="M160 95 L160 155 A 60 60 0 0 1 100 113 Z" fill="var(--c-accent-cyan)" opacity="0.85" />
          <path d="M160 95 L100 113 A 60 60 0 0 1 160 35 Z" fill="var(--c-accent-orange)" opacity="0.85" />
        </SvgContainer>
      );

    case 'Parallel Sets':
    case 'Proportional Area Chart':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'Enterprise' },
            { color: 'var(--c-accent-purple)', label: 'Mid-Market' },
            { color: 'var(--c-accent-emerald)', label: 'SMB' }
          ]} />
          <SvgAxes xLabels={['Small', 'Medium', 'Large']} yTicks={['0', '500']} />
          <rect x="60" y="42" width="60" height="110" rx="6" fill="var(--c-accent-blue)" opacity="0.75" />
          <rect x="140" y="62" width="70" height="90" rx="6" fill="var(--c-accent-purple)" opacity="0.75" />
          <rect x="230" y="92" width="50" height="60" rx="6" fill="var(--c-accent-emerald)" opacity="0.75" />
        </SvgContainer>
      );

    case 'Pie Chart':
    case 'Pie':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={categoryChartData} cx="50%" cy="50%" outerRadius={80} dataKey="value" stroke="none">
              {categoryChartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <RechartsTooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
      );

    case 'Donut Chart':
    case 'Donut':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={categoryChartData} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value" stroke="none">
              {categoryChartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '11px', color: 'var(--c-text-secondary)' }} />
            <RechartsTooltip content={<CustomTooltip />} />
          </PieChart>
        </ResponsiveContainer>
      );

    case 'Treemap':
      return (
        <ResponsiveContainer width="100%" height="100%">
          <Treemap data={treemapData} dataKey="size" aspectRatio={4 / 3} stroke="var(--c-bg)" fill="var(--c-accent-cyan)">
            <RechartsTooltip content={<CustomTooltip />} />
          </Treemap>
        </ResponsiveContainer>
      );

    case 'Word Cloud':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'High Frequency' },
            { color: 'var(--c-accent-emerald)', label: 'Medium Frequency' }
          ]} />
          <text x="160" y="95" textAnchor="middle" fill="var(--c-accent-blue)" fontSize="28" fontWeight="bold">Analytics</text>
          <text x="90" y="55" textAnchor="middle" fill="var(--c-accent-purple)" fontSize="20" fontWeight="600">AI Data</text>
          <text x="230" y="65" textAnchor="middle" fill="var(--c-accent-emerald)" fontSize="18">Metrics</text>
          <text x="90" y="135" textAnchor="middle" fill="var(--c-accent-cyan)" fontSize="16">Insights</text>
          <text x="220" y="145" textAnchor="middle" fill="var(--c-accent-orange)" fontSize="22" fontWeight="bold">Growth</text>
        </SvgContainer>
      );

    // === MAPS / GEOGRAPHICAL ===
    case 'Bubble Map':
    case 'Choropleth Map':
    case 'Connection Map':
    case 'Dot Map':
    case 'Flow Map':
      return (
        <SvgContainer>
          <SvgLegend items={[
            { color: 'var(--c-accent-blue)', label: 'North America Hub' },
            { color: 'var(--c-accent-purple)', label: 'EMEA Hub' }
          ]} />
          
          <line x1="30" y1="160" x2="290" y2="160" stroke="var(--c-glass-border)" strokeWidth="1.5" />
          <line x1="30" y1="25" x2="30" y2="160" stroke="var(--c-glass-border)" strokeWidth="1.5" />
          {['-90°', '0°', '+90°'].map((lat, i) => (
            <text key={i} x="25" y={160 - i * 65} textAnchor="end" fill="var(--c-text-secondary)" fontSize="8">{lat}</text>
          ))}
          {['-180°', '-90°', '0°', '+90°', '+180°'].map((lon, i) => (
            <text key={i} x={30 + i * 65} y="174" textAnchor="middle" fill="var(--c-text-secondary)" fontSize="8">{lon}</text>
          ))}

          <path d="M40 65 Q60 35 90 55 Q120 75 100 125 Q70 145 50 105 Z" fill="var(--c-glass-border)" opacity="0.35" />
          <path d="M160 45 Q210 35 250 65 Q270 105 240 135 Q190 155 150 105 Z" fill="var(--c-glass-border)" opacity="0.35" />
          
          {type === 'Bubble Map' && (
            <>
              <circle cx="75" cy="80" r="14" fill="var(--c-accent-blue)" opacity="0.75" />
              <circle cx="190" cy="75" r="22" fill="var(--c-accent-purple)" opacity="0.75" />
              <circle cx="220" cy="115" r="10" fill="var(--c-accent-emerald)" opacity="0.75" />
            </>
          )}

          {type === 'Dot Map' && (
            <>
              {[70, 85, 90, 170, 190, 205, 220, 230].map((cx, i) => (
                <circle key={i} cx={cx} cy={55 + (i % 4) * 20} r="4" fill="var(--c-accent-cyan)" />
              ))}
            </>
          )}

          {type === 'Connection Map' || type === 'Flow Map' ? (
            <>
              <circle cx="75" cy="80" r="5" fill="var(--c-accent-blue)" />
              <circle cx="190" cy="75" r="5" fill="var(--c-accent-purple)" />
              <circle cx="220" cy="115" r="5" fill="var(--c-accent-emerald)" />
              <path d="M75 80 Q130 35 190 75" fill="none" stroke="var(--c-accent-blue)" strokeWidth="2" strokeDasharray="4 4" />
              <path d="M190 75 Q210 95 220 115" fill="none" stroke="var(--c-accent-purple)" strokeWidth="2" />
            </>
          ) : null}

          {type === 'Choropleth Map' && (
            <>
              <path d="M40 65 Q60 35 90 55 Z" fill="var(--c-accent-blue)" opacity="0.8" />
              <path d="M160 45 Q210 35 210 85 Z" fill="var(--c-accent-purple)" opacity="0.8" />
            </>
          )}
        </SvgContainer>
      );

    default:
      return (
        <SvgContainer>
          <circle cx="160" cy="95" r="40" fill="var(--c-accent-blue)" opacity="0.6" />
        </SvgContainer>
      );
  }
};
