// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using System.ComponentModel;

namespace OnnxExtensions.Models;

/// <summary>
/// onnx 模型的任务类型
/// </summary>
public enum ModelTaskDef
{
    /// <summary>
    /// 检测
    /// </summary>
    Detect = 0,

    /// <summary>
    /// 分割
    /// </summary>
    Segment,

    /// <summary>
    /// 分类
    /// </summary>
    Classify,

    /// <summary>
    /// 姿势估计
    /// </summary>
    Pose,

    /// <summary>
    /// 定向对象检测
    /// </summary>
    OrientedBoundingBoxes,
}
