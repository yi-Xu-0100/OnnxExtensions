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

            float width = x2 - x1;
            float height = y2 - y1;

            var label = model.Labels[classId];
            var rect = new Rectangle((int)x1, (int)y1, (int)width, (int)height);
            result.Add(new(label, rect, score));
        }

        return result;
    }


    private static List<ModelPrediction> ParseOrientedBoundingBoxes(
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

            float width = x2 - x1;
            float height = y2 - y1;

            var label = model.Labels[classId];
            var rect = new Rectangle((int)x1, (int)y1, (int)width, (int)height);
            result.Add(new(label, rect, score));
        }

        return result;
    }

}
