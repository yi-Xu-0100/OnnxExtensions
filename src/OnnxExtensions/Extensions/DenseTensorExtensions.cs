// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace OnnxExtensions.Extensions;

/// <summary>
/// DenseTensor 的扩展方法
/// </summary>
public static class DenseTensorExtensions
{
    /// <summary>
    /// 将 OpenCvSharp Mat 转换为 3 通道 <see cref="DenseTensor&lt;float&gt;"/>
    /// 输出 shape: [1, 3, height, width]，值归一化到 [0,1]
    /// 支持灰度图和 BGR 彩色图
    /// </summary>
    public static unsafe DenseTensor<float> To3ChDenseTensorFloat(Mat image)
    {
        int width = image.Width;
        int height = image.Height;
        int channels = image.Channels();

        var tensor = new DenseTensor<float>(new[] { 1, 3, height, width });

        byte* ptr = image.DataPointer;

        int step = (int)image.Step(); // 每行字节数

        if (channels == 3)
        {
            // 彩色 BGR
            for (int y = 0; y < height; y++)
            {
                byte* rowPtr = ptr + y * step;
                for (int x = 0; x < width; x++)
                {
                    byte b = rowPtr[x * 3 + 0];
                    byte g = rowPtr[x * 3 + 1];
                    byte r = rowPtr[x * 3 + 2];

                    tensor[0, 0, y, x] = r / 255f; // R
                    tensor[0, 1, y, x] = g / 255f; // G
                    tensor[0, 2, y, x] = b / 255f; // B
                }
            }
        }
        else if (channels == 1)
        {
            // 灰度图
            for (int y = 0; y < height; y++)
            {
                byte* rowPtr = ptr + y * step;
                for (int x = 0; x < width; x++)
                {
                    float normalized = rowPtr[x] / 255f;
                    tensor[0, 0, y, x] = normalized;
                    tensor[0, 1, y, x] = normalized;
                    tensor[0, 2, y, x] = normalized;
                }
            }
        }
        else
        {
            throw new ArgumentException("不支持的通道数：" + channels);
        }

        return tensor;
    }
}
