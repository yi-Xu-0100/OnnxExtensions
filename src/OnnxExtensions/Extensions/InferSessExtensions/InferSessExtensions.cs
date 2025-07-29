// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using System.Drawing;
using Microsoft.ML.OnnxRuntime;
using OnnxExtensions.Models;

namespace OnnxExtensions.Extensions;

/// <summary>
/// InferenceSession 类的辅助方法。
/// </summary>
public static partial class InferSessExtensions
{
    /// <summary>
    /// 转换到 Onnx 模型类
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns></returns>
    public static OnnxModel ToOnnxModel(this InferenceSession session)
    {
        OnnxModel onnxModel = new(session.InputDimensions(), session.OutputDimensions(), session.Outputs(), []);

        Dictionary<int, string> classes = session.ParseNames();

        for (int i = 0; i < classes.Count; i++)
        {
            onnxModel.Labels.Add(new(i, classes[i], GetEnumerableColor((uint)i)));
        }
        return onnxModel;
    }

    /// <summary>
    /// 获取枚举颜色
    /// 0-7 的索引将返回预定义的颜色。
    /// 超出范围的索引将循环使用这些颜色。
    /// 0: Blue, 1: Red, 2: Green, 3: Yellow,
    /// 4: Purple, 5: Orange, 6: Cyan, 7: Magenta
    /// </summary>
    /// <param name="index">颜色索引</param>
    /// <returns>颜色</returns>
    private static Color GetEnumerableColor(uint index)
    {
        Color[] colors =
        [
            Color.Blue,
            Color.Red,
            Color.Green,
            Color.Yellow,
            Color.Purple,
            Color.Orange,
            Color.Cyan,
            Color.Magenta,
        ];

        return colors[index & 7];
    }
}
