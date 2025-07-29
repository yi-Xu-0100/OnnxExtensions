// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using Microsoft.ML.OnnxRuntime;

namespace OnnxExtensions.Extensions;

public static partial class InferSessionExtensions
{

    /// <summary>
    /// 获取模型输出信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>模型输出信息</returns>
    public static string[] Outputs(this InferenceSession session)
    {
        return [.. session.OutputMetadata.Keys];
    }

    /// <summary>
    /// 获取模型输出维度信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>模型输入维度信息</returns>
    public static int[] OutputDimensions(this InferenceSession session)
    {
        return session.OutputMetadata[session.OutputMetadata.Keys.First()].Dimensions;
    }
}
