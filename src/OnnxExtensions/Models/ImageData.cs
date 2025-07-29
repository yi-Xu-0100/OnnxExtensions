// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

namespace OnnxExtensions.Models;

/// <summary>
/// 图像数据模型
/// 用于存储图像的宽度、高度、通道数、位深度、像素类型和原始数据。
/// 适用于图像处理和机器学习任务。
/// </summary>
public class ImageData
{
    /// <summary>
    /// 图像宽度
    /// </summary>
    public int Width { get; set; }

    /// <summary>
    /// 图像高度
    /// </summary>
    public int Height { get; set; }

    /// <summary>
    /// 通道数
    /// 1=灰度，3=RGB，4=RGBA
    /// </summary>
    public int Channels { get; set; }

    /// <summary>
    /// 位深度 8, 16, 32等
    /// </summary>
    public int BitDepth { get; set; }

    /// <summary>
    /// 像素类型 "byte", "ushort", "float" 等
    /// </summary>
    public string PixelType { get; set; } = string.Empty;

    /// <summary>
    /// 原始图像数据
    /// </summary>
    public byte[] Data { get; set; } = [];
}
