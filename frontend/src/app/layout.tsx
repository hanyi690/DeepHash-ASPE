import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Navbar from '@/components/Navbar'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'DeepHash-ASPE | 隐私保护跨模态检索',
  description: '基于深度跨模态哈希和非对称标量积保持加密的隐私保护跨模态检索演示系统',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh-CN">
      <body className={`${inter.className} min-h-screen flex flex-col`}>
        <Navbar />
        <main className="flex-1 py-8">
          {children}
        </main>
        <footer className="bg-white/80 backdrop-blur-sm border-t border-gray-200 mt-auto">
          <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
            <div className="flex flex-col md:flex-row justify-between items-center gap-4">
              <p className="text-center text-sm text-gray-500">
                © 2024 DeepHash-ASPE. 隐私保护跨模态检索演示系统
              </p>
              <div className="flex items-center space-x-4 text-sm text-gray-500">
                <span>DCMH + ASPE</span>
                <span>•</span>
                <span>v1.0.0</span>
              </div>
            </div>
          </div>
        </footer>
      </body>
    </html>
  )
}
