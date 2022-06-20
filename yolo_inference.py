import argparse
import os
import warnings

import cv2
import torch
import numpy as np
import rasterio
from rasterio.plot import reshape_as_raster, reshape_as_image
from rasterio.windows import Window
from tqdm import tqdm
from shapely.geometry import Polygon
import geopandas as gpd

warnings.filterwarnings("ignore")

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
COLOR = (255, 0, 0)
THICKNESS = 1
STD_NORMALIZE = 3.5


def get_yolo_predict(model_path, raster_path, output_path, bands_order, step=512, visualize=False, std_norm=3.5,
                     normalize=False):
    print('Model uploading... \n')
    model = torch.hub.load(
        "ultralytics/yolov5:master", "custom", model_path, verbose=True)
    print('Model has been uploaded \n')

    geo_save_path = os.path.join(output_path, 'detected_objects.geojson')
    dst_raster_path = os.path.join(output_path, 'predict.tif')

    src = rasterio.open(raster_path)
    w, h = src.meta['width'], src.meta['height']
    if w < step and h < step:
        step = np.min([w, h])

    whole_rem_w = divmod(w, step)
    whole_rem_h = divmod(h, step)

    all_steps_h = [(0, i * step, 0, step) for i in range(whole_rem_h[0])]

    if whole_rem_h[1] != 0:
        all_steps_h = all_steps_h + [(0, all_steps_h[-1][1] + step, 0, whole_rem_h[1])]

    all_steps = []
    for h_step in all_steps_h:
        all_steps = all_steps + [(i * step, h_step[1], step, h_step[-1]) for i in range(whole_rem_w[0])]
        if whole_rem_w[1] != 0:
            all_steps = all_steps + [(all_steps[-1][0] + step, h_step[1], whole_rem_w[1], h_step[-1])]
    if normalize:
        print('MAX of normalization is being calculated...\n')
        pixels_sum = np.sum([np.sum(src.read(bands_order, window=Window(*i)), axis=(1, 2)) for i in all_steps], axis=0)

        means_channels = (pixels_sum / (w * h)).reshape((3, 1, 1))

        squared_deviation = np.sum(
            [np.sum((src.read(bands_order, window=Window(*i)) - means_channels) ** 2, axis=(1, 2)) for i in all_steps],
            axis=0)

        std = (squared_deviation / (w * h)) ** 0.5

        max_ = (means_channels.reshape(3, -1) + std_norm * std.reshape(3, -1)).reshape(3, 1, 1)
        print('MAX of normalization has been calculated\n')

    profile = src.profile
    profile['dtype'] = 'uint8'
    profile['count'] = 3
    profile['nodata'] = 0
    with rasterio.open(
            dst_raster_path, 'w', **profile
    ) as dst:
        print('Start of predictions...\n')
        detections = []
        for i in tqdm(all_steps):
            x_start, y_start, x_step, y_step = i
            window_normalize = src.read(bands_order, window=Window(x_start, y_start, x_step, y_step))
            if normalize:
                mask_none = np.where(np.sum(window_normalize, axis=0) == 0, True, False)
                window_normalize = np.clip((window_normalize / max_) * 255, 1, 255).astype(rasterio.uint8)
                for channel in range(3):
                    window_normalize[channel][mask_none] = 0

            image = reshape_as_image(np.array(window_normalize)).copy()
            preds = model(image, size=512, augment=True)
            ans = preds.pandas().xyxy[0]
            ans["w"] = (ans.xmax - ans.xmin).astype(int)
            ans["h"] = (ans.ymax - ans.ymin).astype(int)
            ans["x"] = ans.xmin.astype(int)
            ans["y"] = ans.ymin.astype(int)
            for j, pred in ans.iterrows():
                x, y, w, h, confidence, label = pred['x'], pred['y'], pred['w'], pred['h'], pred['confidence'], pred[
                    'name']
                x, y = x + x_start, y + y_start
                coords_box = [(x, y), (x, y + h), (x + w, y + h), (x + w, y)]
                polygon = Polygon([src.transform * box_cs for box_cs in coords_box])
                detections.append({'geometry': polygon, 'label': label})
                if visualize:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (51, 255, 51), 2)

                    cv2.putText(image, label, (x, y), FONT,
                                FONT_SCALE, COLOR, THICKNESS, cv2.LINE_AA)
            if visualize:
                dst.write(reshape_as_raster(image), window=Window(*i))

        gpd.GeoDataFrame(detections, crs=src.meta['crs']).to_file(
            geo_save_path, driver='GeoJSON')
        print(f'Predictions are saved in :  {geo_save_path} \n')

        if visualize:
            print(f'Visualization of predictions  is ready! It is in :  {dst_raster_path} \n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yolo_path", type=str, help="Path to yolo model"
    )
    parser.add_argument(
        "--raster_path", type=str, help="Path to raster"
    )
    parser.add_argument(
        "--output_path", type=str, help="Path to folder where put output"
    )
    parser.add_argument(
        "--normalize", action='store_true', help="whether normalize raster"
    )
    parser.add_argument(
        '--bands_order', nargs='+', type=int, help='bands order in raster  for yolo predict and output'
    )
    parser.add_argument(
        '--step', type=int, default=512, help='size of window for yolo'
    )
    parser.add_argument(
        '--visual', action='store_true', help="whether visual predictions"
    )
    args = parser.parse_args()
    get_yolo_predict(
        args.yolo_path, args.raster_path, args.output_path, tuple(args.bands_order),
        step=args.step, visualize=args.visual, std_norm=STD_NORMALIZE, normalize=args.normalize
    )
