// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using Microsoft.ML.OnnxRuntime;

namespace OnnxExtensions.Extensions;

public static partial class InferSessionExtensions
{
    /// <summary>
    /// 获取模型日期信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns></returns>
    public static string Date(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["date"] ?? string.Empty;
    }

    /// <summary>
    /// 尝试解析日期信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <param name="date">日期</param>
    /// <returns>true: 解析日期成功; false: 解析日期失败</returns>
    public static bool TryParseDate(this InferenceSession session, out DateTime date)
    {
        return DateTime.TryParse(session.ModelMetadata.CustomMetadataMap["date"], out date);
    }
}
