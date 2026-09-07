/**
 * 纪律看板的共享状态。
 *
 * 置顶横幅挂在 `+page.svelte` 的 `<main>` 顶部（全局常驻，切到别的页也看得见——
 * 挂在纪律页内部等于纪律形同虚设），纪律页本身是另一个组件。两者必须看到同一份数据：
 * 在横幅里处置了事件，纪律页要立刻同步，反之亦然。
 *
 * `getDisciplinePositions` 会顺带执行一次扫描（重算止损、回写持仓最高价、落待处理裁决），
 * 所以共享一份还能避免同一次交互里重复扫描。
 */

import { errorMessage, getDisciplinePositions } from '../services';
import type { DisciplineBoard } from '../types';

let board = $state<DisciplineBoard | null>(null);
let loading = $state(false);
let error = $state('');
/** 竞态序号：慢请求的结果不能覆盖后发起的快请求 */
let sequence = 0;

export const discipline = {
    get board() {
        return board;
    },
    get loading() {
        return loading;
    },
    get error() {
        return error;
    },
    get pending() {
        return board?.pending_events ?? [];
    },
    clearError() {
        error = '';
    },
};

export async function refreshDiscipline(silent = false): Promise<void> {
    const current = ++sequence;
    if (!silent) loading = true;
    try {
        const next = await getDisciplinePositions();
        if (current !== sequence) return;
        board = next;
        error = '';
    } catch (e) {
        if (current !== sequence) return;
        error = errorMessage(e, '纪律看板加载失败');
    } finally {
        if (current === sequence) loading = false;
    }
}
