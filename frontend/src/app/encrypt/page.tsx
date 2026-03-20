'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';

export default function EncryptPage() {
  const [step, setStep] = useState(0);

  const steps = [
    {
      title: '原始哈希码',
      description: 'DCMH 生成的 {-1, +1} 哈希码',
      code: '[+1, -1, +1, +1, -1, -1, +1, ...]',
      icon: '📊'
    },
    {
      title: '向量扩展',
      description: '将 d 维向量扩展到 (d+1) 维',
      code: 'p̂ = (p, -0.5||p||²)',
      icon: '📐'
    },
    {
      title: '矩阵变换',
      description: '使用密钥矩阵 M 进行线性变换',
      code: "p' = M^T × p̂",
      icon: '🔄'
    },
    {
      title: '加密完成',
      description: '得到密文向量，保持内积性质',
      code: '[0.234, -1.567, 0.891, ...]',
      icon: '🔐'
    }
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">ASPE 加密过程可视化</h1>
        <p className="text-gray-600">
          非对称标量积保持加密（ASPE）如何保护哈希码隐私
        </p>
      </div>

      {/* Overview Card */}
      <div className="card mb-8">
        <h2 className="text-xl font-semibold mb-4">🔐 ASPE 方案 1 原理</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="font-medium text-blue-900 mb-2">数据库加密</h3>
            <p className="text-sm text-blue-700 mb-2">
              将哈希码扩展到 (d+1) 维，使用 M^T 变换
            </p>
            <code className="text-xs bg-white px-2 py-1 rounded block">
              p̂ = (p, -0.5||p||²)
            </code>
          </div>

          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="font-medium text-green-900 mb-2">查询加密</h3>
            <p className="text-sm text-green-700 mb-2">
              生成随机缩放因子 r，使用 M^(-1) 变换
            </p>
            <code className="text-xs bg-white px-2 py-1 rounded block">
              q̂ = r × (q, 1)
            </code>
          </div>

          <div className="bg-purple-50 p-4 rounded-lg">
            <h3 className="font-medium text-purple-900 mb-2">内积保持</h3>
            <p className="text-sm text-purple-700 mb-2">
              密文内积等于 r 倍原始内积
            </p>
            <code className="text-xs bg-white px-2 py-1 rounded block">
              p̂' · q̂' = r × (p · q)
            </code>
          </div>
        </div>
      </div>

      {/* Step by Step Visualization */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">加密步骤演示</h2>
        <div className="flex flex-wrap justify-center gap-4 mb-8">
          {steps.map((s, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              className={`px-4 py-3 rounded-lg font-medium transition-all ${
                step === i
                  ? 'bg-primary-600 text-white shadow-lg scale-105'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              }`}
            >
              <span className="text-xl mr-2">{s.icon}</span>
              {s.title}
            </button>
          ))}
        </div>

        <motion.div
          key={step}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="card"
        >
          <div className="text-center">
            <div className="text-5xl mb-4">{steps[step].icon}</div>
            <h3 className="text-2xl font-bold mb-2">{steps[step].title}</h3>
            <p className="text-gray-600 mb-6">{steps[step].description}</p>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-lg inline-block">
              {steps[step].code}
            </div>
          </div>
        </motion.div>
      </div>

      {/* Security Properties */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">🛡️ 安全特性</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-start space-x-3">
            <div className="text-green-500 text-xl">✓</div>
            <div>
              <h3 className="font-medium">距离不可恢复</h3>
              <p className="text-sm text-gray-600">
                无法从密文恢复精确的欧几里得距离
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="text-green-500 text-xl">✓</div>
            <div>
              <h3 className="font-medium">陷阱门不可链接</h3>
              <p className="text-sm text-gray-600">
                无法将查询陷阱门与特定查询关联
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="text-green-500 text-xl">✓</div>
            <div>
              <h3 className="font-medium">排序保持</h3>
              <p className="text-sm text-gray-600">
                密文距离排序与明文距离排序一致
              </p>
            </div>
          </div>
          <div className="flex items-start space-x-3">
            <div className="text-green-500 text-xl">✓</div>
            <div>
              <h3 className="font-medium">抵抗已知样本攻击</h3>
              <p className="text-sm text-gray-600">
                2 级安全性，抵抗已知样本攻击
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
