// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using System.Drawing;
using OpenCvSharp;

namespace OnnxExtensions.Models;

/// <summary>
/// 模型预测
/// 用于存储模型预测的标签、矩形区域、面积和置信度。
/// 适用于目标检测和图像分类任务。
/// </summary>
public class ModelPrediction
{
    /// <summary>
    /// 模型标签
    /// 包含预测的类别信息和颜色。
    /// </summary>
    public ModelLabel Label { get; set; }

    /// <summary>
    /// 预测的矩形区域
    /// 用于表示目标在图像中的位置。
    /// </summary>
    public RotatedRect Rectangle { get; set; }

    /// <summary>
    /// 预测的置信度分数
    /// </summary>
    public float Score { get; set; }

    /// <summary>
    /// 初始化 <see cref="ModelPrediction"/> 类.
    /// </summary>
    /// <param name="label">预测的模型标签</param>
    /// <param name="rectangle">预测的矩形区域</param>
    /// <param name="score">预测的置信度分数</param>
    public ModelPrediction(ModelLabel label, RotatedRect rectangle, float score)
    {
        Label = label;
        Score = score;
        Rectangle = rectangle;
    }
}
