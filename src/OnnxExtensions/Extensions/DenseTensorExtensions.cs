// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace OnnxExtensions.Extensions;

/// <summary>
/// DenseTensor 的扩展方法
/// </summary>
public static class DenseTensorExtensions
{
    /// <summary>
    /// 将 <see cref="Image{Rgb24}"/> 转换为 N,C,H,W 格式的 <see cref="DenseTensor{Float}"/>，
    /// 并把像素值归一化到 [0, 1] 区间。
    /// 返回张量的形状为 [1, 3, height, width].
    /// </summary>
    /// <param name="image">待转换的 RGB24 图像。</param>
    /// <returns>形状为 [1, 3, height, width] 的浮点张量。</returns>
    public static DenseTensor<float> To3ChDenseTensorFloat(Image<Rgb24> image)
    {
        int width = image.Width;
        int height = image.Height;

        var tensor = new DenseTensor<float>([1, 3, height, width]);

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < height; y++)
            {
                Span<Rgb24> pixelRow = accessor.GetRowSpan(y);

                for (int x = 0; x < width; x++)
                {
                    Rgb24 pixel = pixelRow[x];

                    tensor[0, 0, y, x] = pixel.R / 255f;
                    tensor[0, 1, y, x] = pixel.G / 255f;
                    tensor[0, 2, y, x] = pixel.B / 255f;
                }
            }
        });

        return tensor;
    }
}
