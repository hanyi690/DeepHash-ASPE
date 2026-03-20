'use client';

import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, LineChart, Line } from 'recharts';

interface MetricsChartProps {
  data?: any[];
  plaintextMap?: number;
  ciphertextMap?: number;
}

export default function MetricsChart({
  data = [],
  plaintextMap = 0.65,
  ciphertextMap = 0.6498,
}: MetricsChartProps) {
  // 模拟数据
  const defaultData = [
    { num_queries: 10, plaintext_map: 0.652, ciphertext_map: 0.652, difference: 0.0001 },
    { num_queries: 20, plaintext_map: 0.648, ciphertext_map: 0.648, difference: 0.0002 },
    { num_queries: 50, plaintext_map: 0.655, ciphertext_map: 0.655, difference: 0.0001 },
    { num_queries: 100, plaintext_map: 0.651, ciphertext_map: 0.651, difference: 0.0002 },
  ];

  const displayData = data.length > 0 ? data : defaultData;

  // 自定义工具提示
  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-white p-3 rounded-lg shadow-lg border border-gray-200">
          <p className="text-sm font-semibold text-gray-900 mb-2">查询数：{label}</p>
          {payload.map((entry: any, index: number) => (
            <div key={index} className="flex items-center justify-between space-x-4 text-sm">
              <span className="flex items-center">
                <span className="w-2 h-2 rounded-full mr-2" style={{ backgroundColor: entry.color }} />
                {entry.name}
              </span>
              <span className="font-mono font-medium">{(entry.value as number).toFixed(4)}</span>
            </div>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="space-y-6">
      {/* mAP 对比 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <svg className="w-5 h-5 text-[#6366F1] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          mAP 对比
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={displayData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="num_queries"
              stroke="#6b7280"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={[0.6, 0.7]}
              stroke="#6b7280"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => value.toFixed(2)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            <Bar
              name="明文 mAP"
              dataKey="plaintext_map"
              fill="#6366F1"
              radius={[4, 4, 0, 0]}
              maxBarSize={40}
            />
            <Bar
              name="密文 mAP"
              dataKey="ciphertext_map"
              fill="#10B981"
              radius={[4, 4, 0, 0]}
              maxBarSize={40}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* 差异趋势 */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
          <svg className="w-5 h-5 text-emerald-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
          差异趋势
        </h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={displayData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              dataKey="num_queries"
              stroke="#6b7280"
              fontSize={12}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              stroke="#6b7280"
              fontSize={12}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => value.toFixed(4)}
            />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ paddingTop: '20px' }} />
            <Line
              name="差异"
              type="monotone"
              dataKey="difference"
              stroke="#8B5CF6"
              strokeWidth={2}
              dot={{ fill: '#8B5CF6', strokeWidth: 2, r: 4 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* 汇总统计 */}
      <div className="grid grid-cols-3 gap-4">
        <div className="card text-center p-4">
          <div className="text-sm text-gray-500 mb-1">平均明文 mAP</div>
          <div className="text-2xl font-bold text-[#6366F1] font-mono">
            {plaintextMap.toFixed(4)}
          </div>
        </div>
        <div className="card text-center p-4">
          <div className="text-sm text-gray-500 mb-1">平均密文 mAP</div>
          <div className="text-2xl font-bold text-emerald-600 font-mono">
            {ciphertextMap.toFixed(4)}
          </div>
        </div>
        <div className="card text-center p-4">
          <div className="text-sm text-gray-500 mb-1">平均差异</div>
          <div className="text-2xl font-bold text-purple-600 font-mono">
            {Math.abs(plaintextMap - ciphertextMap).toFixed(4)}
          </div>
        </div>
      </div>
    </div>
  );
}
