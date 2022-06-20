# multi_detectiion_SS
Repo consists of inference script and inference notebook, which get satellite image and find coordinates of boxes of objects and their labels in image, and next give geojson with coordinates and labels. Also if set the parameter, can get the visualization of the predictions. 
Using model is YoloV4, which findes 8 classes:
          -car
          -truck
          -building
          -boat
          -aircraft
          -vessel
          -railway vehicle
          -engineering vehicle
Model was trained on SkySat and WorldView domain with RGB order of channels. So better predictions will be on the same domains and order.

First of all, install requirements :
```
pip install -r requirements.txt
```
Run script inference:

```
python yolo_inference.py --yolo_path
                         --raster_path
                         --output_path
                         --normalize
                         --bands_order
                         --step
```
