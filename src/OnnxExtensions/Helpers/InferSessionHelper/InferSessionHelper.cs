// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using System.Reflection;
using Microsoft.ML.OnnxRuntime;

namespace OnnxExtensions.Helpers;

/// <summary>
/// InferenceSession 类的辅助方法。
/// 提供加载嵌入的 ONNX 模型的功能。
/// </summary>
public static class InferSessionHelper
{
    /// <summary>
    /// 加载嵌入的 ONNX 模型。
    /// 该方法从嵌入资源中加载 ONNX 模型，并返回一个新的 InferenceSession 实例。
    /// 注意：资源名称应为嵌入的 ONNX 模型文件的完整名称，例如 "OnnxExtensions.Models.Yolo11.onnx"。
    /// 如果需要自定义 SessionOptions，可以通过参数传递。
    /// </summary>
    /// <param name="resourceName">嵌入的 ONNX 模型文件的完整名称，例如 "OnnxExtensions.Models.Yolo11.onnx"</param>
    /// <param name="sessionOptions">会话选项</param>
    /// <returns>新的 InferenceSession 实例</returns>
    /// <exception cref="FileNotFoundException">如果找不到指定的资源，将抛出 FileNotFoundException。</exception>
    public static InferenceSession LoadEmbeddedOnnxModel(string resourceName, Assembly assembly, SessionOptions? sessionOptions = null)
    {
        string[] resourceNames = assembly.GetManifestResourceNames();
        if (!Array.Exists(resourceNames, r => r == resourceName))
        {
            throw new FileNotFoundException($"Not Found Embedded .Onnx Model({resourceName}) in {assembly.GetName().Name}");
        }
        using Stream stream = assembly.GetManifestResourceStream(resourceName)!;

        // 将流复制到内存中（OnnxRuntime 要求可 seek 的 Stream）
        using var memoryStream = new MemoryStream();
        stream.CopyTo(memoryStream);
        memoryStream.Seek(0, SeekOrigin.Begin);

        return new InferenceSession(memoryStream.ToArray(), sessionOptions ?? new());
    }
}
