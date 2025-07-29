// Copyright (c) 2025-now yi-Xu-0100.
// This file is licensed under the MIT License. See LICENSE for details.

using System.Text.Json;
using Microsoft.ML.OnnxRuntime;

namespace OnnxExtensions.Extensions;

public static partial class InferSessionExtensions
{
    /// <summary>
    /// 获取模型图像尺寸信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns></returns>
    // ReSharper disable once MemberCanBePrivate.Global
    public static string ImgSize(this InferenceSession session)
    {
        return session.ModelMetadata.CustomMetadataMap["imgsz"] ?? "[-1, -1]";
    }

    /// <summary>
    /// 尝试解析图像尺寸信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <param name="imgSize">图像尺寸</param>
    /// <returns>true: 解析图像尺寸成功; false: 解析图像尺寸失败</returns>
    public static bool TryParseImgSize(this InferenceSession session, out List<int> imgSize)
    {
        imgSize = [];
        string validJson = session.ImgSize().ToValidJson();
        var result = JsonSerializer.Deserialize<List<int>>(validJson);
        if (result == null)
        {
            return false;
        }
        imgSize = result;
        return true;
    }

    /// <summary>
    /// 解析图像尺寸信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>图像尺寸 [width, height]</returns>
    public static List<int> ParseImgSize(this InferenceSession session)
    {
        string validJson = session.ImgSize().ToValidJson();
        var result = JsonSerializer.Deserialize<List<int>>(validJson);
        if (result == null)
        {
            throw new InvalidDataException("Invalid image size");
        }
        return result;
    }

    /// <summary>
    /// 尝试解析图像宽度信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <param name="imgWidth">图像宽度</param>
    /// <returns>true: 解析图像宽度成功; false: 解析图像宽度失败</returns>
    public static bool TryParseImgWidth(this InferenceSession session, out int imgWidth)
    {
        imgWidth = -1;
        string validJson = session.ImgSize().ToValidJson();
        var result = JsonSerializer.Deserialize<List<int>>(validJson);
        if (result == null)
        {
            return false;
        }
        imgWidth = result[0];
        return true;
    }

    /// <summary>
    /// 解析图像宽度信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>图像宽度</returns>
    public static int ParseImgWidth(this InferenceSession session)
    {
        string validJson = session.ImgSize().ToValidJson();
        var result = JsonSerializer.Deserialize<List<int>>(validJson);
        if (result == null)
        {
            throw new InvalidDataException("Invalid image size");
        }
        return result[0];
    }

    /// <summary>
    /// 尝试解析图像高度信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <param name="imgHeight">图像高度</param>
    /// <returns>true: 解析图像高度成功; false: 解析图像高度失败</returns>
    public static bool TryParseImgHeight(this InferenceSession session, out int imgHeight)
    {
        imgHeight = -1;
        string validJson = session.ImgSize().ToValidJson();
        var result = JsonSerializer.Deserialize<List<int>>(validJson);
        if (result == null)
        {
            return false;
        }
        imgHeight = result[1];
        return true;
    }

    /// <summary>
    /// 解析图像高度信息。
    /// </summary>
    /// <param name="session">模型连接</param>
    /// <returns>图像高度</returns>
    public static int ParseImgHeight(this InferenceSession session)
    {
        string validJson = session.ImgSize().ToValidJson();
        var result = JsonSerializer.Deserialize<List<int>>(validJson);
        if (result == null)
        {
            throw new InvalidDataException("Invalid image size");
        }
        return result[1];
    }
}
