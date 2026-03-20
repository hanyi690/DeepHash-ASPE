'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

export default function EncryptPage() {
  const [step, setStep] = useState(0);

  const steps = [
    {
      title: '原始哈希码',
      description: 'DCMH 生成的 {-1, +1} 哈希码',
      code: '[+1, -1, +1, +1, -1, -1, +1, ...]',
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      )
    },
    {
      title: '向量扩展',
      description: '将 d 维向量扩展到 (d+1) 维',
      code: 'p̂ = (p, -0.5||p||²)',
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1h-2a1 1 0 01-1-1v-4z" />
        </svg>
      )
    },
    {
      title: '矩阵变换',
      description: '使用密钥矩阵 M 进行线性变换',
      code: "p' = M^T × p̂",
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
        </svg>
      )
    },
    {
      title: '加密完成',
      description: '得到密文向量，保持内积性质',
      code: '[0.234, -1.567, 0.891, ...]',
      icon: (
        <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
      )
    }
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* 页面标题 */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">ASPE 加密过程可视化</h1>
        <p className="text-gray-600">
          非对称标量积保持加密（ASPE）如何保护哈希码隐私
        </p>
      </div>

      {/* 原理概览 */}
      <div className="card mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <svg className="w-5 h-5 text-[#6366F1] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          ASPE 方案 1 原理
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-50/80 border border-blue-100 p-4 rounded-xl">
            <div className="flex items-center mb-2">
              <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center mr-3">
                <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
                </svg>
              </div>
              <h3 className="font-semibold text-blue-900">数据库加密</h3>
            </div>
            <p className="text-sm text-blue-700 mb-3">
              将哈希码扩展到 (d+1) 维，使用 M^T 变换
            </p>
            <code className="text-xs bg-white/80 px-2 py-1.5 rounded-lg block font-mono text-blue-800">
              p̂ = (p, -0.5||p||²)
            </code>
          </div>

          <div className="bg-emerald-50/80 border border-emerald-100 p-4 rounded-xl">
            <div className="flex items-center mb-2">
              <div className="w-8 h-8 bg-emerald-500 rounded-lg flex items-center justify-center mr-3">
                <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z" />
                </svg>
              </div>
              <h3 className="font-semibold text-emerald-900">查询加密</h3>
            </div>
            <p className="text-sm text-emerald-700 mb-3">
              生成随机缩放因子 r，使用 M^(-1) 变换
            </p>
            <code className="text-xs bg-white/80 px-2 py-1.5 rounded-lg block font-mono text-emerald-800">
              q̂ = r × (q, 1)
            </code>
          </div>

          <div className="bg-purple-50/80 border border-purple-100 p-4 rounded-xl">
            <div className="flex items-center mb-2">
              <div className="w-8 h-8 bg-purple-500 rounded-lg flex items-center justify-center mr-3">
                <svg className="w-4 h-4 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                </svg>
              </div>
              <h3 className="font-semibold text-purple-900">内积保持</h3>
            </div>
            <p className="text-sm text-purple-700 mb-3">
              密文内积等于 r 倍原始内积
            </p>
            <code className="text-xs bg-white/80 px-2 py-1.5 rounded-lg block font-mono text-purple-800">
              p̂&apos; · q̂&apos; = r × (p · q)
            </code>
          </div>
        </div>
      </div>

      {/* 步骤演示 */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <svg className="w-5 h-5 text-[#6366F1] mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          加密步骤演示
        </h2>

        {/* 步骤切换器 */}
        <div className="flex flex-wrap justify-center gap-3 mb-6">
          {steps.map((s, i) => (
            <button
              key={i}
              onClick={() => setStep(i)}
              className={`px-4 py-2.5 rounded-xl font-medium transition-all duration-200 cursor-pointer
                ${step === i
                  ? 'bg-[#6366F1] text-white shadow-lg shadow-[#6366F1]/25 scale-105'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
            >
              <div className="flex items-center space-x-2">
                <span className={step === i ? 'text-white' : 'text-gray-400'}>
                  {s.icon}
                </span>
                <span className="hidden sm:inline">{s.title}</span>
                <span className="sm:hidden">步骤 {i + 1}</span>
              </div>
            </button>
          ))}
        </div>

        {/* 步骤内容 */}
        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.2 }}
            className="card"
          >
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-gradient-to-br from-[#6366F1] to-[#818CF8] rounded-2xl flex items-center justify-center text-white">
                {steps[step].icon}
              </div>
              <h3 className="text-2xl font-bold text-gray-900 mb-2">{steps[step].title}</h3>
              <p className="text-gray-600 mb-6">{steps[step].description}</p>
              <div className="bg-gray-900 text-emerald-400 p-5 rounded-xl font-mono text-base inline-block w-full max-w-md overflow-x-auto">
                {steps[step].code}
              </div>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* 安全特性 */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4 flex items-center">
          <svg className="w-5 h-5 text-emerald-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          安全特性
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="flex items-start space-x-3 p-3 bg-gray-50 rounded-xl">
            <div className="w-6 h-6 bg-emerald-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <h3 className="font-medium text-gray-900">距离不可恢复</h3>
              <p className="text-sm text-gray-600 mt-1">
                无法从密文恢复精确的欧几里得距离
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3 p-3 bg-gray-50 rounded-xl">
            <div className="w-6 h-6 bg-emerald-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <h3 className="font-medium text-gray-900">陷阱门不可链接</h3>
              <p className="text-sm text-gray-600 mt-1">
                无法将查询陷阱门与特定查询关联
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3 p-3 bg-gray-50 rounded-xl">
            <div className="w-6 h-6 bg-emerald-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <h3 className="font-medium text-gray-900">排序保持</h3>
              <p className="text-sm text-gray-600 mt-1">
                密文距离排序与明文距离排序一致
              </p>
            </div>
          </div>

          <div className="flex items-start space-x-3 p-3 bg-gray-50 rounded-xl">
            <div className="w-6 h-6 bg-emerald-500 rounded-full flex items-center justify-center flex-shrink-0 mt-0.5">
              <svg className="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <div>
              <h3 className="font-medium text-gray-900">抵抗已知样本攻击</h3>
              <p className="text-sm text-gray-600 mt-1">
                2 级安全性，抵抗已知样本攻击
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
