// 添加成交量格式化函数
export function formatVolume(volume: number): string {
  if (volume >= 1_000_000_000)
    return `${(volume / 1_000_000_000).toFixed(2)}亿`;
  if (volume >= 10_000) return `${(volume / 10_000).toFixed(2)}万`;
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

// 格式化涨跌额
export function formatChange(change: number): string {
  if (change === 0) return "0.00";
  return change > 0 ? `+${change.toFixed(2)}` : change.toFixed(2);
}

// 格式化涨跌幅
export function formatChangePercent(changePercent: number): string {
  if (changePercent === 0) return "0.00%";
  return changePercent > 0 ? `+${changePercent.toFixed(2)}%` : `${changePercent.toFixed(2)}%`;
}

// 格式化价格（保留2位小数）
export function formatPrice(price: number): string {
  return price.toFixed(2);
}

// 格式化百分比（保留2位小数）
export function formatPercent(percent: number): string {
  return `${percent.toFixed(2)}%`;
}
