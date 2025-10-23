// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

namespace OnnxExtensions.Models;

/// <summary>
/// ONNX 模型
/// </summary>
public class OnnxModel
{
    /// <summary>
    /// 任务类型
    /// </summary>
    public ModelTaskDef TaskDef { get; }

    /// <summary>
    /// 输入维度
    /// </summary>
    public int[] InputDimensions { get; }

    /// <summary>
    /// 输出维度
    /// </summary>
    public int[] OutputDimensions { get; }

    /// <summary>
    /// 输出名称
    /// </summary>
    public string[] Outputs { get; }

    /// <summary>
    /// 标签
    /// </summary>
    public List<ModelLabel> Labels { get; }

    /// <summary>
    /// 初始化 <see cref="OnnxModel"/> 类.
    /// </summary>
    /// <param name="inputDimensions">输入维度.</param>
    /// <param name="outputDimensions">输出维度.</param>
    /// <param name="outputs">输出名称.</param>
    /// <param name="labels">标签.</param>
    public OnnxModel(int[] inputDimensions, int[] outputDimensions, string[] outputs, List<ModelLabel> labels, ModelTaskDef taskDef)
    {
        InputDimensions = inputDimensions;
        OutputDimensions = outputDimensions;
        Outputs = outputs;
        Labels = labels;
        TaskDef = taskDef;
    }
}
