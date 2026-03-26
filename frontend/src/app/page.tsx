'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';

export default function Home() {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="text-center py-20"
      >
        {/* Logo/图标 */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="flex justify-center mb-6"
        >
          <div className="w-20 h-20 bg-gradient-to-br from-[#6366F1] to-[#818CF8] rounded-2xl flex items-center justify-center shadow-lg">
            <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
        </motion.div>

        <h1 className="text-5xl md:text-6xl font-bold text-gray-900 mb-4">
          <span className="text-gradient">DeepHash-ASPE</span>
        </h1>
        <p className="text-xl text-gray-600 mb-10 max-w-2xl mx-auto leading-relaxed">
          基于深度跨模态哈希和非对称标量积保持加密的
          <br className="hidden md:block" />
          隐私保护跨模态检索演示系统
        </p>

        {/* CTA 按钮 */}
        <Link href="/search">
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            className="btn-primary px-10 py-4 text-lg shadow-lg shadow-[#6366F1]/25 hover:shadow-[#6366F1]/40 hover:-translate-y-0.5 transition-all duration-200"
          >
            <svg className="w-5 h-5 mr-2 inline" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            开始检索
          </motion.button>
        </Link>
      </motion.div>

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card"
        >
          <div className="w-12 h-12 bg-[#6366F1]/10 rounded-xl flex items-center justify-center mb-4">
            <svg className="w-6 h-6 text-[#6366F1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">隐私保护</h3>
          <p className="text-gray-600 leading-relaxed">
            使用 ASPE 加密技术，确保检索过程中的数据隐私安全，抵抗已知样本攻击
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card"
        >
          <div className="w-12 h-12 bg-emerald-500/10 rounded-xl flex items-center justify-center mb-4">
            <svg className="w-6 h-6 text-emerald-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">跨模态检索</h3>
          <p className="text-gray-600 leading-relaxed">
            支持标签搜图和图搜标签的双向跨模态检索，64 位哈希码高效匹配
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card"
        >
          <div className="w-12 h-12 bg-purple-500/10 rounded-xl flex items-center justify-center mb-4">
            <svg className="w-6 h-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-900 mb-2">图搜图</h3>
          <p className="text-gray-600 leading-relaxed">
            基于 CNN 特征的图像检索，支持 SkNN 隐私保护检索模式
          </p>
        </motion.div>
      </div>

      {/* 技术特点 */}
      <div className="py-12">
        <h2 className="text-2xl font-bold text-gray-900 mb-8 text-center">技术特点</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <svg className="w-5 h-5 text-[#6366F1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              DCMH 深度跨模态哈希
            </h3>
            <ul className="text-gray-600 space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-[#6366F1] mt-1">•</span>
                端到端学习图像和文本的统一哈希表示
              </li>
              <li className="flex items-start gap-2">
                <span className="text-[#6366F1] mt-1">•</span>
                支持 Flickr25K 和 NUS-WIDE 数据集
              </li>
              <li className="flex items-start gap-2">
                <span className="text-[#6366F1] mt-1">•</span>
                64 位紧凑哈希码实现高效检索
              </li>
            </ul>
          </div>
          <div className="card">
            <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center gap-2">
              <svg className="w-5 h-5 text-[#6366F1]" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              ASPE 加密方案
            </h3>
            <ul className="text-gray-600 space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-[#6366F1] mt-1">•</span>
                非对称标量积保持加密保护数据隐私
              </li>
              <li className="flex items-start gap-2">
                <span className="text-[#6366F1] mt-1">•</span>
                加密前后检索精度损失远小于 0.001
              </li>
              <li className="flex items-start gap-2">
                <span className="text-[#6366F1] mt-1">•</span>
                抵抗已知样本攻击
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}