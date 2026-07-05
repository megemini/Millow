# Millow

> **Millow** — **M**（代表 MoonBit）+ **illow**（致敬 Python 的 [Pillow](https://python-pillow.org/)）。

一个**零 FFI、跨平台**的 MoonBit 图像处理库。`millow` 完全基于内存中的
RGBA8 缓冲区（`Array[Byte]`，布局为 `H × W × 4`）工作，支持所有后端：
`wasm-gc`、`wasm`、`js` 和 `native`。

[English](README.md) | 中文

## 演示

`cmd/main` 包含轻量级演示，生成合成渐变图像并应用 30+ 种 millow 操作 — 无需外部依赖：

```
moon run cmd/main
```

完整演示（PNG/JPEG I/O，30 张输出图像）位于独立的 `examples/` 项目中：

```
cd examples
moon run .
```

| 输入 | `to_grayscale` | `tint(100,150,200)` | `gaussian_blur(σ=2)` |
|:---:|:---:|:---:|:---:|
| <img src="docs/images/lena_ai_generated.png" width="200"> | <img src="docs/images/demo_grayscale.jpg" width="200"> | <img src="docs/images/demo_tint.jpg" width="200"> | <img src="docs/images/demo_gaussian_blur.jpg" width="200"> |

| `sharpen(1.0)` | `sobel` | `equalize_histogram` | `threshold_otsu` |
|:---:|:---:|:---:|:---:|
| <img src="docs/images/demo_sharpen.jpg" width="200"> | <img src="docs/images/demo_sobel.jpg" width="200"> | <img src="docs/images/demo_equalize_histogram.jpg" width="200"> | <img src="docs/images/demo_threshold_otsu.jpg" width="200"> |

| `rotate_any(45°)` | `find_contours` | `pipeline` |
|:---:|:---:|:---:|
| <img src="docs/images/demo_rotate_45.jpg" width="200"> | <img src="docs/images/demo_contours.jpg" width="200"> | <img src="docs/images/demo_pipeline.jpg" width="200"> |

## 功能特性

- **核心图像类型** — `Image`，支持构造、像素访问、克隆、通道分离/合并、子图像。
- **颜色** — 灰度（平坦与加权）、反色、色调、BGR、Alpha 展平、HSV/YCbCr 转换、
  LUT 应用、Alpha 合成。
- **几何变换** — 裁剪、翻转、90/180/270 旋转、任意角度旋转、平移、仿射变换、
  错切、缩放（最近邻/双线性/双三次）、等比缩放、适配/覆盖、缩略图、填充。
- **增强** — 亮度、对比度、伽马、归一化、自动对比度、标准化、锐化、Unsharp Mask。
- **阈值与直方图** — 固定阈值、Otsu、Sauvola、直方图（灰度与彩色）、均衡化、
  CLAHE、直方图匹配。
- **滤波与边缘** — 卷积、均值/高斯模糊、中值/最小/最大值滤波、双边滤波、
  Sobel、Scharr、Prewitt、Laplacian、Canny。
- **形态学** — 腐蚀、膨胀、开运算、闭运算、形态学梯度、Top-hat、Black-hat、
  骨架化、击中击不中。
- **特征检测** — LBP、HOG、Harris 角点、Shi-Tomasi 角点。
- **测量** — 连通域、轮廓提取、矩、Hu 矩、区域属性、像素计数。
- **数据增强** — 随机裁剪、翻转、旋转、亮度/对比度/伽马调整、高斯/椒盐噪声、
  色彩抖动、可组合的增强管线与加权随机选择。
- **度量** — MSE、PSNR、SSIM。
- **绘图** — 像素、直线、矩形、圆、椭圆、多边形、泛洪填充。
- **I/O** — PPM/PGM 序列化，可插拔的 `Encoder`/`Decoder` 注册表。
- **边界处理** — `Replicate`（复制）、`Reflect`（镜像）、`Wrap`（平铺）、
  `Constant`（常量填充）四种模式，用于仿射变换和插值。

## 项目结构

```
millow/
├── src/              # 实现包（megemini/millow/src）
├── millow.mbt        # 根门面：重新导出公共 API（megemini/millow）
├── test/             # 黑盒测试包，测试公共 API
├── test_alignment/   # 与 Python（skimage/Pillow）参考实现对齐测试
├── cmd/main/         # 轻量级合成图像演示（仅依赖 millow）
└── examples/         # 独立完整演示（通过 mizchi/image 进行 PNG/JPEG I/O）
```

根包是 `src` 的薄**门面**，下游用户只需 `import "megemini/millow"`，
即可通过 `@millow` 访问全部 API。

## 安装

```
moon add megemini/millow
```

然后在包的 `moon.pkg` 中导入：

```
import {
  "megemini/millow" @millow,
}
```

## 快速开始

```mbt nocheck
///|
test "构建、变换并检查图像" {
  // 64×64 画布上绘制一个填充矩形
  let base = Image::from_pixel(64, 64, 30, 60, 90, 255)
  let canvas = draw_rect(base, 8, 8, 40, 40, 220, 40, 40, 255, true, 1)

  // 灰度 → 模糊 → 边缘检测
  let gray = to_grayscale(canvas)
  let blurred = gaussian_blur(gray, 1.5)
  let edges = sobel(blurred)
  assert_eq(edges.shape(), (64, 64))

  // Otsu 自适应阈值
  let (_, binary) = threshold_otsu(blurred)
  assert_eq(binary.shape(), (64, 64))

  // 缩小并序列化为 PPM
  let thumb = resize(canvas, 16, 16, Nearest)
  let ppm = to_ppm(thumb)
  assert_eq(ppm[0], 'P'.to_int().to_byte())
}
```

> 上面的示例中 API 以非限定名调用，因为它们在 `millow` 包内部运行。
> 从其他模块调用时，需加上导入别名前缀，例如 `@millow.to_grayscale(img)`。

## 增强管线

使用 `augment_pipeline` 将多个增强操作组合成一次调用，按顺序依次应用每个
`Augmentation` 变体：

```mbt nocheck
///|
test "augment_pipeline 示例" {
  let img = Image::from_pixel(64, 64, 30, 60, 90, 255)
  let out = augment_pipeline(img, [
    FlipHorizontal,
    Rotate(15.0),
    Brightness(1.2),
    Contrast(1.3),
    NoiseGaussian(8.0),
  ])
  assert_true(out.h > 0 && out.w > 0)
}
```

可用的 `Augmentation` 变体：`Crop(y, x, h, w)`、`Resize(dst_h, dst_w)`、
`FlipHorizontal`、`FlipVertical`、`Rotate(angle)`、`Brightness(factor)`、
`Contrast(factor)`、`Gamma(g)`、`NoiseGaussian(std)`、`NoiseSaltPepper(prob)`、
`ColorJitter(b, c, s, h)`。当裁剪/缩放参数无效时 `augment_pipeline` 会抛出异常。

使用 `augment_random_choice` 从加权分布中采样一个增强操作：

```mbt nocheck
///|
test "augment_random_choice 示例" {
  let img = Image::from_pixel(64, 64, 30, 60, 90, 255)
  let out = augment_random_choice(img, [
    (0.4, FlipVertical),
    (0.4, Gamma(0.8)),
    (0.2, ColorJitter(0.2, 0.2, 0.0, 0.0)),
  ])
  assert_eq(out.shape(), img.shape())
}
```

## API 说明

### 坐标系

millow 在整个 API 中使用统一的坐标约定：

- **维度顺序**：`(h, w)` — 高度优先，宽度次之。
  `Image::new(h, w)`、`resize(img, dst_h, dst_w, interp)`、
  `crop(img, y, x, h, w)`、`Image::shape() -> (h, w)`。

- **坐标顺序**：`(y, x)` — 行优先，列次之。
  `y` 是垂直轴（向下递增），`x` 是水平轴（向右递增）。原点 `(0, 0)` 为
  **左上角**。

- **绘图中心**：`draw_circle(img, cy, cx, radius, ...)`、
  `draw_ellipse(img, cy, cx, ry, rx, ...)`。

- **平移**：`translate(img, dy, dx, interp)`。

- **轮廓**：`find_contours` 返回 `(y, x)` 元组。

### 亮度调整

`adjust_brightness(img, factor)` 使用乘法因子：

- `factor = 1.0` 返回原图
- `factor = 0.0` 返回全黑图像
- 大于 1.0 的值提亮图像
- 小于 1.0 的值压暗图像

### 对比度调整

`adjust_contrast(img, factor)` 相对于图像平均亮度调整对比度：

- `factor = 1.0` 返回原图
- `factor = 0.0` 返回等于图像平均亮度的纯灰图像
- 大于 1.0 的值增加对比度
- 小于 1.0 的值降低对比度

### Alpha 合成

`flatten_alpha(img, r, g, b)` 将图像合成到指定纯色背景上，使用浮点混合以获得平滑效果。

### 边界处理

部分操作支持 `mode` 参数，控制边界像素的处理方式：

- `Replicate` — 将最近的边缘像素向外延伸
- `Reflect` — 沿边缘镜像像素
- `Wrap` — 周期性平铺图像
- `Constant(r, g, b, a)` — 用指定颜色填充边界区域

支持可选 `mode` 参数的函数包括 `affine_transform` 和 `shear`。
大多数其他操作内部使用 replicate（钳位）边界处理。

### 双边滤波

`bilateral_filter(img, d, sigma_color, sigma_space)` 应用保边平滑：

- `d` 为像素邻域直径（传 0 则根据 `sigma_space` 自动计算）
- `sigma_color` 控制颜色相似度阈值（越大平滑越多）
- `sigma_space` 控制空间邻近度阈值（越大邻域越宽）

### 仿射变换

`affine_transform(img, matrix, dst_h, dst_w, interp, mode)` 使用 6 元素矩阵
`[a, b, c, d, e, f]` 应用通用仿射变换，表示：

```
x' = a*x + b*y + c
y' = d*x + e*y + f
```

常用变换可使用 `rotate_any` 和 `translate`。

### 随机噪声

`random_noise_gaussian(img, std)` 添加指定标准差的高斯噪声。

`random_noise_salt_pepper(img, amount)` 添加指定比例（受影响像素占比）的椒盐噪声。

## 后端

`millow` 不包含任何外部函数调用。已在 `wasm-gc`、`wasm`、`js`、`native` 上
验证构建，测试套件在所有后端均通过。

## 测试

```
moon test                 # 运行所有测试
moon test --target native # 指定后端
moon run cmd/main         # 运行合成演示（无外部依赖）
cd examples && moon run . # 运行完整演示（含 JPEG I/O）
```

### 对齐测试

`test_alignment/` 验证 millow 的输出与 Python 参考实现（numpy / skimage / Pillow）
一致。工作流程如下：

1. `test_alignment/generate_fixtures.py` 为小测试图像计算期望输出字节，
   写入 `fixtures_test.mbt` 作为 `Array[Int]` 字面量。
2. MoonBit 测试从这些 fixture 构造 `Image`，运行每个 millow 操作，
   然后逐字节比较（整数操作精确匹配，浮点操作 ±1 容差）。

修改算法后重新生成 fixture：

```
source $HOME/venv310/bin/activate
python test_alignment/generate_fixtures.py
moon test
```

## 路线图

完整的版本规划与未来计划请参见 [docs/roadmap.md](docs/roadmap.md)。

## 许可证

Apache-2.0。详见 [LICENSE](LICENSE)。
