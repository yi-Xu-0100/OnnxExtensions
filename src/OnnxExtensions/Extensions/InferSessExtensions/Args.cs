// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using System.Text.Json;
using Microsoft.ML.OnnxRuntime;

namespace OnnxExtensions.Extensions;

public static partial class InferSessionExtensions
{
    /// <summary>
    /// 获取模型参数信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns></returns>
    // ReSharper disable once MemberCanBePrivate.Global
    public static string Args(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["args"] ?? string.Empty;
    }

    /// <summary>
    /// 尝试解析模型参数字典。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <param name="args">模型参数字典</param>
    /// <returns>true: 解模型参数字典成功; false: 解析模型参数字典失败</returns>
    public static bool TryParseArgs(this InferenceSession session, out Dictionary<string, string> args)
    {
        args = [];
        string validJson = session.Args().ToValidJson();
        var result = JsonSerializer.Deserialize<Dictionary<string, string>>(validJson);
        if (result == null)
        {
            return false;
        }
        args = result;
        return true;
    }

    /// <summary>
    /// 解析模型参数字典信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>模型参数字典</returns>
    public static Dictionary<string, string> ParseArgs(this InferenceSession session)
    {
        string validJson = session.Args().ToValidJson();
        var result = JsonSerializer.Deserialize<Dictionary<string, string>>(validJson);
        if (result == null)
        {
            throw new InvalidDataException("Invalid image size");
        }
        return result;
    }
}
