// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using OnnxExtensions.Models;

namespace OnnxExtensions.Helpers;

/// <summary>
/// Yolo 模型的辅助方法
/// </summary>
public static class YoloHelper
{
    public static List<ModelPrediction> ParseOutput(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        Image<Rgb24> image,
        float conf,
        OnnxModel model
    )
    {
        string firstOutput = model.Outputs[0];
        DenseTensor<float> output = (DenseTensor<float>)outputs.First(x => x.Name == firstOutput).Value;
        return ParseDetect(output, image, conf, model);
    }

    private static List<ModelPrediction> ParseDetect(
        DenseTensor<float> output,
        Image<Rgb24> image,
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
