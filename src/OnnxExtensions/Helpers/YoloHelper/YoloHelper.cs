// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using System.Drawing;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxExtensions.Models;
using OpenCvSharp;

namespace OnnxExtensions.Helpers;

/// <summary>
/// Yolo 模型的辅助方法
/// </summary>
public static class YoloHelper
{
    /// <summary>
    /// 解析 ONNX 模型的输出，并将结果转换为对应的预测列表。
    /// </summary>
    /// <param name="outputs">ONNX 推理后的输出集合，每个元素为 <see cref="DisposableNamedOnnxValue"/>。</param>
    /// <param name="image">用于推理的输入图像，类型为 <see cref="Mat"/>，用于将输出映射回图像坐标。</param>
    /// <param name="conf">置信度阈值，低于该值的预测将被过滤。</param>
    /// <param name="model">ONNX 模型信息，包括输出名称和任务类型。</param>
    /// <returns>返回解析后的预测列表，每个预测为 <see cref="ModelPrediction"/> 类型。</returns>
    /// <exception cref="NotSupportedException">当模型任务类型不被支持时抛出。</exception>
    public static List<ModelPrediction> ParseOutput(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        Mat image,
        float conf,
        OnnxModel model
    )
    {
        DenseTensor<float> output = (DenseTensor<float>)outputs.First(x => x.Name == model.Outputs[0]).Value;
        return model.TaskDef switch
        {
            ModelTaskDef.Detect => ParseDetect(output, image, conf, model),
            ModelTaskDef.OrientedBoundingBoxes => ParseOrientedBoundingBoxes(output, image, conf, model),
            _ => throw new NotSupportedException($"模型类型[{model.TaskDef}]暂不支持!")
        };
    }

    private static List<ModelPrediction> ParseDetect(
        DenseTensor<float> output,
        Mat image,
        float confThreshold,
        OnnxModel model
    )
    {
        List<ModelPrediction> result = [];

        // [1, 300, 6]
        ReadOnlySpan<int> dims = output.Dimensions;
        int numDetections = dims[1];

        float imageWidth = image.Width;
        float imageHeight = image.Height;

        for (int i = 0; i < numDetections; i++)
        {
            float x1 = output[0, i, 0];
            float y1 = output[0, i, 1];
            float x2 = output[0, i, 2];
            float y2 = output[0, i, 3];
            float score = output[0, i, 4];
            int classId = (int)output[0, i, 5];

            if (score < confThreshold || classId >= model.Labels.Count)
                continue;

            // 确保坐标在图像范围内
            x1 = Math.Clamp(x1, 0, imageWidth);
            y1 = Math.Clamp(y1, 0, imageHeight);
            x2 = Math.Clamp(x2, 0, imageWidth);
            y2 = Math.Clamp(y2, 0, imageHeight);

            var center = new Point2f((x1 + x2) /2, (y1 + y2) /2);

            float width = x2 - x1;
            float height = y2 - y1;
            var size = new Size2f(width, height);

            var label = model.Labels[classId];
            var rect = new RotatedRect(center, size, 0);
            result.Add(new(label, rect, score));
        }

        return result;
    }


    /// <summary>
    /// 解析 YOLO Oriented Bounding Box 输出
    /// </summary>
    private static List<ModelPrediction> ParseOrientedBoundingBoxes(
        DenseTensor<float> output,
        Mat image,
        float confThreshold,
        OnnxModel model)
    {
        List<ModelPrediction> result = new();

        ReadOnlySpan<int> dims = output.Dimensions;
        int numDetections = dims[1];

        float imgW = image.Width;
        float imgH = image.Height;

        for (int i = 0; i < numDetections; i++)
        {
            float cx = output[0, i, 0];
            float cy = output[0, i, 1];
            float w = output[0, i, 2];
            float h = output[0, i, 3];
            float score = output[0, i, 4];
            int classId = (int)output[0, i, 5];
            float angle = output[0, i, 6]; // 单位通常是度或弧度，需确认

            if (score < confThreshold || classId >= model.Labels.Count)
                continue;

            // 坐标范围裁剪
            // cx = Math.Clamp(cx, 0, imgW);
            // cy = Math.Clamp(cy, 0, imgH);
            // w = Math.Max(1, Math.Min(w, imgW));
            // h = Math.Max(1, Math.Min(h, imgH));

            // 创建旋转矩形
            var center = new Point2f(cx, cy);
            var size = new Size2f(w, h);

            // YOLO-OBB 输出角度一般是弧度（部分实现是度）
            // 如果发现旋转角度不对，可改成 angle * 180f / MathF.PI
            var rotatedRect = new RotatedRect(center, size, angle* 180f / MathF.PI);

            // 四点坐标
            Point2f[] vertices = rotatedRect.Points();

            var label = model.Labels[classId];

            // 可根据需要自定义 ModelPrediction
            result.Add(new (label, rotatedRect, score));
        }

        return result;
    }


}
