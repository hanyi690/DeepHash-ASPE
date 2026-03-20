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

  return (
    <div className="space-y-6">
      {/* mAP 对比 */}
      <div>
        <h3 className="text-lg font-semibold mb-4">mAP 对比</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={displayData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="num_queries" />
            <YAxis domain={[0.6, 0.7]} />
            <Tooltip />
            <Legend />
            <Bar name="明文 mAP" dataKey="plaintext_map" fill="#3b82f6" />
            <Bar name="密文 mAP" dataKey="ciphertext_map" fill="#22c55e" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* 差异趋势 */}
      <div>
        <h3 className="text-lg font-semibold mb-4">差异趋势</h3>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={displayData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="num_queries" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line name="差异" type="monotone" dataKey="difference" stroke="#a855f7" strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* 汇总统计 */}
      <div className="grid grid-cols-3 gap-4 pt-4 border-t">
        <div className="text-center">
          <div className="text-sm text-gray-500 mb-1">平均明文 mAP</div>
          <div className="text-2xl font-bold text-blue-600">
            {(plaintextMap).toFixed(4)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-sm text-gray-500 mb-1">平均密文 mAP</div>
          <div className="text-2xl font-bold text-green-600">
            {(ciphertextMap).toFixed(4)}
          </div>
        </div>
        <div className="text-center">
          <div className="text-sm text-gray-500 mb-1">平均差异</div>
          <div className="text-2xl font-bold text-purple-600">
            {Math.abs(plaintextMap - ciphertextMap).toFixed(4)}
          </div>
        </div>
      </div>
    </div>
  );
}
