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
      {searchTime !== undefined && (
        <div className="mb-4 text-sm text-gray-600">
          检索完成，耗时 {searchTime.toFixed(2)} ms，共 {results.length} 个结果
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {results.map((result, index) => (
          <div
            key={index}
            className="card hover:shadow-lg transition-shadow"
          >
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-gray-500">
                #{result.rank} - 图像 {result.image_id}
              </span>
              <span className="text-xs bg-primary-100 text-primary-700 px-2 py-1 rounded-full">
                距离：{result.distance.toFixed(2)}
              </span>
            </div>

            <div className="bg-gray-100 rounded-lg h-40 mb-3 flex items-center justify-center overflow-hidden">
              {result.thumbnail_url ? (
                <img
                  src={result.thumbnail_url}
                  alt={`结果 ${result.rank}`}
                  className="w-full h-full object-cover"
                />
              ) : (
                <span className="text-gray-400">图像 {result.image_id}</span>
              )}
            </div>

            <div className="flex items-center justify-between mb-2">
              <span className="text-xs text-gray-500">相似度</span>
              <span className="text-sm font-medium text-green-600">
                {(result.score * 100).toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-green-500 h-2 rounded-full transition-all"
                style={{ width: `${result.score * 100}%` }}
              />
            </div>

            {result.captions && result.captions.length > 0 && (
              <div className="mt-3 pt-3 border-t">
                <p className="text-xs text-gray-500 mb-2">相关描述:</p>
                <ul className="text-sm text-gray-700 space-y-1">
                  {result.captions.slice(0, 2).map((caption, i) => (
                    <li key={i} className="flex items-start">
                      <span className="text-gray-400 mr-2 text-xs">•</span>
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
