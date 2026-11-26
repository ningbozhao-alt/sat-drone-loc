# SatDroneLoc

在 Windows 上本地运行。在态势地图上画出大约 5 公里见方的任务区，用在线或离线卫星瓦片生成搜索底图，再上传一张近垂直航拍图，匹配出图像中心、四角范围，以及照片上点击位置的经纬度。

匹配在 CPU 上完成。地物清楚、卫星图和航拍季节接近时，中心点误差预期是数米到几十米。夜航、厚云、大片农田或水面、卫星图过期时会失败，并且不会在地图上放置确定点。

## 运行

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器打开 http://127.0.0.1:8000 。

也可以双击项目目录里的 `启动.bat`。服务没开时它会启动并打开浏览器；已经在运行时只打开浏览器。

## 自检

建立搜索底图后，点击「用底图自检」。程序从镶嵌图中心裁一块约 800 像素的图，并把它旋转 90 度再定位一次。两次中心误差都应小于 3 个镶嵌图像素。一张无关图片应被拒绝。

命令行：

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 瓦片

默认使用 Esri World Imagery，并缓存到 `data/tiles/online`。国内网络较慢时，在页面里把瓦片 URL 改成带 `{z}`、`{x}`、`{y}` 的地址，例如天地图影像（`tk` 自行申请）：

```text
https://t0.tianditu.gov.cn/DataServer?T=img_w&x={x}&y={y}&l={z}&tk=你的密钥
```

也可以设置环境变量 `SATDRONE_TILE_URL`。

离线来源：

- XYZ 目录，结构为 `{z}/{x}/{y}.png`
- MBTiles
- GeoTIFF（需要 rasterio）

页面上的「已缓存」只读取已经下载过的瓦片，不再访问网络。

## 不做的事

全球无先验盲搜、连续视频、惯导融合、大倾角斜视、目标检测。默认标出的是图像中心对应的地面点；照片上的点击点是配准之后的像素反算。
