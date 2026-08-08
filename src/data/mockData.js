export const timeSeriesData = [
  { name: 'Jan', value: 4000, secondary: 2400 },
  { name: 'Feb', value: 3000, secondary: 1398 },
  { name: 'Mar', value: 5000, secondary: 9800 },
  { name: 'Apr', value: 4500, secondary: 3908 },
  { name: 'May', value: 6000, secondary: 4800 },
  { name: 'Jun', value: 7200, secondary: 3800 },
  { name: 'Jul', value: 8500, secondary: 4300 },
];

export const categoricalData = [
  { name: 'Enterprise', value: 400 },
  { name: 'SMB', value: 300 },
  { name: 'Consumer', value: 300 },
  { name: 'Government', value: 200 },
];

export const scatterData = [
  { x: 100, y: 200, z: 200 },
  { x: 120, y: 100, z: 260 },
  { x: 170, y: 300, z: 400 },
  { x: 140, y: 250, z: 280 },
  { x: 150, y: 400, z: 500 },
  { x: 110, y: 280, z: 200 },
];

export const radarData = [
  { subject: 'Math', A: 120, B: 110, fullMark: 150 },
  { subject: 'Chinese', A: 98, B: 130, fullMark: 150 },
  { subject: 'English', A: 86, B: 130, fullMark: 150 },
  { subject: 'Geography', A: 99, B: 100, fullMark: 150 },
  { subject: 'Physics', A: 85, B: 90, fullMark: 150 },
  { subject: 'History', A: 65, B: 85, fullMark: 150 },
];

export const treemapData = [
  {
    name: 'Technology',
    children: [
      { name: 'Software', size: 1300 },
      { name: 'Hardware', size: 800 },
      { name: 'Services', size: 600 }
    ]
  },
  {
    name: 'Marketing',
    children: [
      { name: 'Digital', size: 900 },
      { name: 'Print', size: 300 }
    ]
  }
];

export const funnelData = [
  { value: 100, name: 'Impressions', fill: 'var(--c-accent-blue)' },
  { value: 80, name: 'Clicks', fill: 'var(--c-accent-cyan)' },
  { value: 50, name: 'Sign Ups', fill: 'var(--c-accent-purple)' },
  { value: 20, name: 'Purchases', fill: 'var(--c-accent-emerald)' },
];

export const composedForecastData = [
  { name: 'Q1', actual: 4000, forecast: 4000, lower: 4000, upper: 4000 },
  { name: 'Q2', actual: 4500, forecast: 4500, lower: 4500, upper: 4500 },
  { name: 'Q3', actual: null, forecast: 5000, lower: 4600, upper: 5400 },
  { name: 'Q4', actual: null, forecast: 5500, lower: 4800, upper: 6200 },
];
