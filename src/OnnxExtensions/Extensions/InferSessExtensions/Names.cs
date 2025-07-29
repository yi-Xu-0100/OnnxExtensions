// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using System.Text.Json;
using Microsoft.ML.OnnxRuntime;

namespace OnnxExtensions.Extensions;

public static partial class InferSessionExtensions
{
    /// <summary>
    /// 获取模型分类信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns></returns>
    // ReSharper disable once MemberCanBePrivate.Global
    public static string Names(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["names"] ?? string.Empty;
    }

    /// <summary>
    /// 尝试解析模型分类字典。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <param name="names">模型分类字典</param>
    /// <returns>true: 解模型分类字典成功; false: 解析模型分类字典失败</returns>
    public static bool TryParseNames(this InferenceSession session, out Dictionary<int, string> names)
    {
        names = [];
        string validJson = session.Names().ToValidJson();
        var result = JsonSerializer.Deserialize<Dictionary<int, string>>(validJson);
        if (result == null)
        {
            return false;
        }
        names = result;
        return true;
    }

    /// <summary>
    /// 解析模型分类字典信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>模型分类字典</returns>
    public static Dictionary<int, string> ParseNames(this InferenceSession session)
    {
        string validJson = session.Names().ToValidJson();
        var result = JsonSerializer.Deserialize<Dictionary<int, string>>(validJson);
        if (result == null)
        {
            throw new InvalidDataException("Invalid image size");
        }
        return result;
    }
}
