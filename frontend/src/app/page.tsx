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
        className="text-center py-16"
      >
        <h1 className="text-5xl font-bold text-gray-900 mb-4">
          DCMH + ASPE 演示系统
        </h1>
        <p className="text-xl text-gray-600 mb-8">
          基于深度跨模态哈希和非对称标量积保持加密的<br/>隐私保护跨模态检索演示系统
        </p>
        <div className="flex justify-center space-x-4">
          <Link href="/demo">
            <button className="btn-primary px-8 py-3 text-lg">
              开始演示
            </button>
          </Link>
          <Link href="/dataset">
            <button className="btn-secondary px-8 py-3 text-lg">
              浏览数据集
            </button>
          </Link>
        </div>
      </motion.div>

      {/* Features */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 py-12">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card"
        >
          <div className="text-center">
            <div className="text-4xl mb-4">🔐</div>
            <h3 className="text-lg font-semibold mb-2">隐私保护</h3>
            <p className="text-gray-600">
              使用 ASPE 加密技术，确保检索过程中的数据隐私安全
            </p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card"
        >
          <div className="text-center">
            <div className="text-4xl mb-4">🔍</div>
            <h3 className="text-lg font-semibold mb-2">跨模态检索</h3>
            <p className="text-gray-600">
              支持文本→图像和图像→文本的跨模态检索功能
            </p>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card"
        >
          <div className="text-center">
            <div className="text-4xl mb-4">📊</div>
            <h3 className="text-lg font-semibold mb-2">性能对比</h3>
            <p className="text-gray-600">
              实时展示明文与密文检索的 mAP 指标对比
            </p>
          </div>
        </motion.div>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 py-8">
        <Link href="/encrypt">
          <div className="card hover:shadow-lg transition-shadow cursor-pointer">
            <h3 className="text-lg font-semibold mb-2">🔒 加密过程可视化</h3>
            <p className="text-gray-600">
              了解 ASPE 如何加密哈希码并保持检索能力
            </p>
          </div>
        </Link>

        <Link href="/metrics">
          <div className="card hover:shadow-lg transition-shadow cursor-pointer">
            <h3 className="text-lg font-semibold mb-2">📈 mAP 性能指标</h3>
            <p className="text-gray-600">
              查看系统检索性能评估和对比分析
            </p>
          </div>
        </Link>
      </div>
    </div>
  );
}
