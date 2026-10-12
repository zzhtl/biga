import { invokeCommand } from './core';
import type { AiConnectionConfig, AiJob } from '../types/ai';

export const listAiConnections = () => invokeCommand<AiConnectionConfig[]>('list_ai_connections');
export const saveAiConnection = (config: AiConnectionConfig) => invokeCommand<AiConnectionConfig>('save_ai_connection', { config });
export const deleteAiConnection = (id: string) => invokeCommand<void>('delete_ai_connection', { id });
export const checkAiConnection = (config: AiConnectionConfig) => invokeCommand<string>('check_ai_connection', { config });
export const startAiPrediction = (symbol: string, connectionId: string, days: number, regenerate = false) =>
  invokeCommand<AiJob>('start_ai_prediction', { request: { symbol, connection_id: connectionId, days, regenerate } });
export const cancelAiPrediction = (id: string) => invokeCommand<void>('cancel_ai_prediction', { id });
export const getAiPrediction = (id: string) => invokeCommand<AiJob>('get_ai_prediction', { id });
export const listAiPredictions = (symbol: string | null = null) => invokeCommand<AiJob[]>('list_ai_predictions', { symbol });
