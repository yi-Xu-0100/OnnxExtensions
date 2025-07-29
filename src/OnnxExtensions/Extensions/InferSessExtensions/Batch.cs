// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using Microsoft.ML.OnnxRuntime;

namespace OnnxExtensions.Extensions;

public static partial class InferSessExtensions
{
    /// <summary>
    /// 获取模型批量信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns></returns>
    public static string Batch(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["batch"] ?? "-1";
    }

    /// <summary>
    /// 尝试解析批量信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <param name="batch">批量</param>
    /// <returns>true: 解析批量成功; false: 解析批量失败</returns>
    public static bool TryParseBatch(this InferenceSession session, out int batch)
    {
        return int.TryParse(session.ModelMetadata.CustomMetadataMap["batch"], out batch);
    }
}
