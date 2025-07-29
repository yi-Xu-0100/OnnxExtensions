// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using System.Text.RegularExpressions;

namespace OnnxExtensions.Extensions;

/// <summary>
/// 用于处理 Json 的扩展方法。
/// </summary>
internal static partial class JsonExtensions
{
    /// <summary>
    /// 将不合法的 Json 字符串转为合法的字符串。
    /// </summary>
    /// <param name="json">不合法的 Json 字符串</param>
    /// <returns></returns>
    internal static string ToValidJson(this string json)
    {
        // 1. 替换 key：如果 key 没引号，用下面这句
        string invalidJson = RegexGeneration.AddDoubleQuotes().Replace(json, "\"$1\":");

        // 2. 替换单引号包裹的字符串 value 为双引号
        return RegexGeneration.ReplaceDoubleQuotes().Replace(invalidJson, "\"$1\"");
    }
}

internal static partial class RegexGeneration
{
    [GeneratedRegex(@"(?<=[,{])\s*(\w+)\s*:")]
    internal static partial Regex AddDoubleQuotes();

    [GeneratedRegex(@"'([^']*)'")]
    internal static partial Regex ReplaceDoubleQuotes();
}
