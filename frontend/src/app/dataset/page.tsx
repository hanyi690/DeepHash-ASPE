'use client';

import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function DatasetPage() {
  const [stats, setStats] = useState<any>(null);
  const [images, setImages] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDatasetInfo();
  }, []);

  const loadDatasetInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/metrics/status`);
      if (response.data.success) {
        setStats(response.data.coco_status);
      }
    } catch (err) {
      console.error('加载数据集信息失败:', err);
    } finally {
      setLoading(false);
    }
  };

  // 模拟数据集图像
  const sampleImages = [
    { id: 1, caption: 'A cat sitting on a chair' },
    { id: 2, caption: 'A dog playing in the park' },
    { id: 3, caption: 'A beautiful sunset over the ocean' },
    { id: 4, caption: 'People walking on the street' },
    { id: 5, caption: 'A bird flying in the sky' },
    { id: 6, caption: 'A car parked on the road' },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">MS-COCO 数据集</h1>
        <p className="text-gray-600">
          MS-COCO 2014 数据集用于跨模态检索演示
        </p>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="card">
          <div className="text-sm text-gray-500 mb-1">图像数量</div>
          <div className="text-3xl font-bold text-primary-600">
            {stats?.num_images || loading ? (stats?.num_images || '-') : '100'}
          </div>
        </div>

        <div className="card">
          <div className="text-sm text-gray-500 mb-1">注释数量</div>
          <div className="text-3xl font-bold text-primary-600">
            {stats?.num_annotations || loading ? (stats?.num_annotations || '-') : '500'}
          </div>
        </div>

        <div className="card">
          <div className="text-sm text-gray-500 mb-1">每张图像描述数</div>
          <div className="text-3xl font-bold text-primary-600">
            {stats?.captions_per_image || loading ? (stats?.captions_per_image || '-') : '5'}
          </div>
        </div>

        <div className="card">
          <div className="text-sm text-gray-500 mb-1">平均描述长度</div>
          <div className="text-3xl font-bold text-primary-600">
            {stats?.avg_caption_length?.toFixed(1) || loading ? (stats?.avg_caption_length?.toFixed(1) || '-') : '8.5'}
          </div>
        </div>
      </div>

      {/* Dataset Info */}
      <div className="card mb-8">
        <h2 className="text-xl font-semibold mb-4">关于 MS-COCO</h2>
        <p className="text-gray-700 mb-4">
          MS-COCO (Microsoft Common Objects in Context) 是一个大规模的图像标注数据集，
          包含超过 330,000 张图像，其中 200,000 张带有标注。每张图像都有 5 个独立的描述性标题，
          使其成为跨模态检索研究的理想数据集。
        </p>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h3 className="font-medium mb-2">数据集特点</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• 真实场景图像</li>
              <li>• 多标签标注</li>
              <li>• 每张图像 5 个标题</li>
              <li>• 80 个物体类别</li>
            </ul>
          </div>
          <div>
            <h3 className="font-medium mb-2">检索配置</h3>
            <ul className="text-sm text-gray-600 space-y-1">
              <li>• 哈希码维度：64</li>
              <li>• 加密方案：ASPE Scheme 1</li>
              <li>• 距离度量：汉明距离</li>
              <li>• 评估指标：mAP</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Sample Images */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">数据集样本</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sampleImages.map((img) => (
            <div key={img.id} className="border rounded-lg overflow-hidden">
              <div className="bg-gray-200 h-40 flex items-center justify-center">
                <span className="text-gray-400">图像 {img.id}</span>
              </div>
              <div className="p-3 bg-gray-50">
                <p className="text-sm text-gray-700">{img.caption}</p>
              </div>
            </div>
          ))}
        </div>
        <p className="text-sm text-gray-500 mt-4 text-center">
          注：演示环境下使用模拟数据展示
        </p>
      </div>
    </div>
  );
}
