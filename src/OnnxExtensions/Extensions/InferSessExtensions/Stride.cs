// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using Microsoft.ML.OnnxRuntime;

namespace OnnxExtensions.Extensions;

public static partial class InferSessExtensions
{
    /// <summary>
    /// 获取模型步长信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns></returns>
    public static string Stride(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["stride"] ?? "-1";
    }

    /// <summary>
    /// 尝试解析步长信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <param name="stride">步长</param>
    /// <returns>true: 解析步长成功; false: 解析步长失败</returns>
    public static bool TryParseStride(this InferenceSession session, out int stride)
    {
        return int.TryParse(session.ModelMetadata.CustomMetadataMap["stride"], out stride);
    }
}
