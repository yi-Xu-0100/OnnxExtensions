// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using Microsoft.ML.OnnxRuntime;

namespace OnnxExtensions.Extensions;

public static partial class InferSessionExtensions
{
    /// <summary>
    /// 获取模型版本信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>模型版本信息</returns>
    public static string Version(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["version"] ?? "Unknown";
    }

    /// <summary>
    /// 获取模型描述信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>模型描述信息</returns>
    public static string Description(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["description"] ?? "No description available.";
    }

    /// <summary>
    /// 获取模型作者信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>模型作者信息</returns>
    public static string Author(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["author"] ?? "Unknown";
    }

    /// <summary>
    /// 获取模型任务信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>模型任务信息</returns>
    public static string Task(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["task"] ?? "Unknown";
    }

    /// <summary>
    /// 获取模型协议信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>模型协议信息</returns>
    public static string License(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["license"] ?? "Unknown";
    }

    /// <summary>
    /// 获取模型文档信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>模型文档信息</returns>
    public static string Docs(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["docs"] ?? "No docs available.";
    }
}
