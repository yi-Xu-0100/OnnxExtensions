// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using Microsoft.ML.OnnxRuntime;

namespace OnnxExtensions.Extensions;

public static partial class InferSessionExtensions
{
    /// <summary>
    /// 获取模型输入图片尺寸信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>模型输入图片尺寸信息</returns>
    public static int[] InputDimensions(this InferenceSession session)
    {
        return session.InputMetadata["images"].Dimensions;
    }
}
