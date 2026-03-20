'use client';

interface SearchResult {
  rank: number;
  image_id: number;
  score: number;
  distance: number;
  captions: string[];
  thumbnail_url?: string;
}

interface ResultsGridProps {
  results: SearchResult[];
  searchTime?: number;
}

export default function ResultsGrid({ results, searchTime }: ResultsGridProps) {
  if (results.length === 0) {
    return null;
  }

  return (
    <div className="w-full">
      {/* 搜索结果统计 */}
      {searchTime !== undefined && (
        <div className="mb-6 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
            <span className="text-sm text-gray-600">
              检索完成，耗时 <span className="font-mono font-semibold text-[#6366F1]">{searchTime.toFixed(2)}</span> ms
            </span>
          </div>
          <span className="badge-primary">
            共 {results.length} 个结果
          </span>
        </div>
      )}

      {/* 结果网格 */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {results.map((result, index) => (
          <div
            key={index}
            className="card group cursor-pointer hover:shadow-lg hover:-translate-y-1 transition-all duration-200"
          >
            {/* 头部：排名和图像 ID */}
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center space-x-2">
                <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                  ${result.rank === 1 ? 'bg-[#6366F1] text-white' : 'bg-gray-100 text-gray-600'}`}>
                  {result.rank}
                </span>
                <span className="text-sm font-medium text-gray-500">
                  图像 {result.image_id}
                </span>
              </div>
              <span className="badge-neutral font-mono text-xs">
                距离：{result.distance.toFixed(2)}
              </span>
            </div>

            {/* 图像区域 */}
            <div className="relative bg-gray-100 rounded-xl h-40 mb-3 overflow-hidden">
              {result.thumbnail_url ? (
                <img
                  src={result.thumbnail_url}
                  alt={`结果 ${result.rank}`}
                  className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-300"
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <svg className="w-12 h-12 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  <span className="text-gray-400 ml-2">图像 {result.image_id}</span>
                </div>
              )}
              {/* 悬停覆盖层 */}
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none" />
            </div>

            {/* 相似度进度条 */}
            <div className="mb-3">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs text-gray-500">相似度</span>
                <span className={`text-sm font-bold font-mono
                  ${result.score >= 0.8 ? 'text-emerald-600' : result.score >= 0.6 ? 'text-[#6366F1]' : 'text-amber-600'}`}>
                  {(result.score * 100).toFixed(1)}%
                </span>
              </div>
              <div className="progress-bar">
                <div
                  className={`progress-bar-fill ${
                    result.score >= 0.8 ? 'from-emerald-500 to-emerald-400' :
                    result.score >= 0.6 ? 'from-[#6366F1] to-[#818CF8]' :
                    'from-amber-500 to-amber-400'
                  }`}
                  style={{ width: `${Math.min(result.score * 100, 100)}%` }}
                />
              </div>
            </div>

            {/* 描述 */}
            {result.captions && result.captions.length > 0 && (
              <div className="pt-3 border-t border-gray-100">
                <p className="text-xs text-gray-500 mb-2 flex items-center">
                  <svg className="w-3 h-3 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
                  </svg>
                  相关描述
                </p>
                <ul className="space-y-1.5">
                  {result.captions.slice(0, 2).map((caption, i) => (
                    <li key={i} className="flex items-start text-sm text-gray-600">
                      <span className="text-[#6366F1] mr-2 mt-1 flex-shrink-0">
                        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      </span>
                      <span className="line-clamp-2">{caption}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
