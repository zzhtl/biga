// 添加成交量格式化函数
export function formatVolume(volume: number): string {
  if (volume >= 1_000_000_000)
    return `${(volume / 1_000_000_000).toFixed(2)}亿`;
  if (volume >= 10_000) return `${(volume / 1_0000).toFixed(2)}万`;
  return volume.toString();
}

// 新增日期格式化工具函数
export function formatDate(dateString: string): string {
  const date = new Date(dateString);

  // 使用UTC时间避免时区问题
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");

  return `${year}-${month}-${day}`;
}
