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
      {/* 页面标题 */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">MS-COCO 数据集</h1>
        <p className="text-gray-600">
          MS-COCO 2014 数据集用于跨模态检索演示
        </p>
      </div>

      {/* 统计卡片 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <div className="card text-center p-5">
          <div className="w-10 h-10 mx-auto mb-3 bg-[#6366F1]/10 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-[#6366F1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <div className="text-sm text-gray-500 mb-1">图像数量</div>
          <div className="text-3xl font-bold text-[#6366F1] font-mono">
            {stats?.num_images || loading ? (stats?.num_images || '-') : '100'}
          </div>
        </div>

        <div className="card text-center p-5">
          <div className="w-10 h-10 mx-auto mb-3 bg-emerald-500/10 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <div className="text-sm text-gray-500 mb-1">注释数量</div>
          <div className="text-3xl font-bold text-emerald-600 font-mono">
            {stats?.num_annotations || loading ? (stats?.num_annotations || '-') : '500'}
          </div>
        </div>

        <div className="card text-center p-5">
          <div className="w-10 h-10 mx-auto mb-3 bg-purple-500/10 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
            </svg>
          </div>
          <div className="text-sm text-gray-500 mb-1">每张图像描述数</div>
          <div className="text-3xl font-bold text-purple-600 font-mono">
            {stats?.captions_per_image || loading ? (stats?.captions_per_image || '-') : '5'}
          </div>
        </div>

        <div className="card text-center p-5">
          <div className="w-10 h-10 mx-auto mb-3 bg-amber-500/10 rounded-xl flex items-center justify-center">
            <svg className="w-6 h-6 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div className="text-sm text-gray-500 mb-1">平均描述长度</div>
          <div className="text-3xl font-bold text-amber-600 font-mono">
            {stats?.avg_caption_length?.toFixed(1) || loading ? (stats?.avg_caption_length?.toFixed(1) || '-') : '8.5'}
          </div>
        </div>
      </div>

      {/* 数据集信息 */}
      <div className="card mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <svg className="w-5 h-5 text-[#6366F1] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          关于 MS-COCO
        </h2>
        <p className="text-gray-700 mb-6 leading-relaxed">
          MS-COCO (Microsoft Common Objects in Context) 是一个大规模的图像标注数据集，
          包含超过 330,000 张图像，其中 200,000 张带有标注。每张图像都有 5 个独立的描述性标题，
          使其成为跨模态检索研究的理想数据集。
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-gray-50 rounded-xl p-4">
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
              <svg className="w-4 h-4 text-[#6366F1] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              数据集特点
            </h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                真实场景图像
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                多标签标注
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                每张图像 5 个标题
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                80 个物体类别
              </li>
            </ul>
          </div>
          <div className="bg-gray-50 rounded-xl p-4">
            <h3 className="font-semibold text-gray-900 mb-3 flex items-center">
              <svg className="w-4 h-4 text-[#6366F1] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
              </svg>
              检索配置
            </h3>
            <ul className="space-y-2 text-sm text-gray-600">
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                哈希码维度：64
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                加密方案：ASPE Scheme 1
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                距离度量：汉明距离
              </li>
              <li className="flex items-start">
                <span className="w-1.5 h-1.5 bg-[#6366F1] rounded-full mt-1.5 mr-2 flex-shrink-0" />
                评估指标：mAP
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* 样本图像 */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <svg className="w-5 h-5 text-[#6366F1] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          数据集样本
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {sampleImages.map((img) => (
            <div key={img.id} className="group border border-gray-200 rounded-xl overflow-hidden hover:border-[#6366F1]/40 hover:shadow-md transition-all duration-200">
              <div className="bg-gradient-to-br from-gray-100 to-gray-200 h-36 flex items-center justify-center">
                <svg className="w-12 h-12 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
              <div className="p-3 bg-white">
                <p className="text-sm text-gray-700 font-medium line-clamp-2">{img.caption}</p>
                <p className="text-xs text-gray-400 mt-1">Image #{img.id}</p>
              </div>
            </div>
          ))}
        </div>
        <p className="text-sm text-gray-500 mt-6 text-center bg-gray-50 py-2 rounded-lg">
          注：演示环境下使用模拟数据展示
        </p>
      </div>
    </div>
  );
}
