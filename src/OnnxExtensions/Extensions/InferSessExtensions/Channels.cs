// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using Microsoft.ML.OnnxRuntime;

namespace OnnxExtensions.Extensions;

public static partial class InferSessionExtensions
{
    /// <summary>
    /// 获取模型通道信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns></returns>
    public static string Channels(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["channels"] ?? string.Empty;
    }

    /// <summary>
    /// 尝试解析通道信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <param name="channels">通道</param>
    /// <returns>true: 解析通道成功; false: 解析通道失败</returns>
    public static bool TryParseChannels(this InferenceSession session, out int channels)
    {
        return int.TryParse(session.ModelMetadata.CustomMetadataMap["channels"], out channels);
    }
}
