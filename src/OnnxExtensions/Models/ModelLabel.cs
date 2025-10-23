// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.


using System.Drawing;

namespace OnnxExtensions.Models;

/// <summary>
/// 模型标签
/// </summary>
public class ModelLabel
{
    /// <summary>
    /// 模型标签ID
    /// </summary>
    public int Id { get; set; }

    /// <summary>
    /// 模型标签名称
    /// </summary>
    public string Name { get; set; }

    /// <summary>
    /// 模型标签颜色
    /// </summary>
    public Color Color { get; set; }

    /// <summary>
    /// 初始化 <see cref="ModelLabel"/> 类.
    /// </summary>
    /// <param name="id">模型标签ID.</param>
    /// <param name="name">模型标签名称.</param>
    /// <param name="color">模型标签颜色.</param>
    public ModelLabel(int id, string name, Color color)
    {
        Id = id;
        Name = name;
        Color = color;
    }

    /// <summary>
    /// 返回模型标签的字符串表示形式.
    /// </summary>
    /// <returns>模型标签的字符串</returns>
    public override string ToString()
    {
        return $"{Id}({Name}) - {Color}";
    }
}
