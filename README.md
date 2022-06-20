# multi_detectiion_SS
Repo consists of inference script and inference notebook, which get satellite image and find coordinates of <br />
boxes of objects and their labels in image, and next give geojson with coordinates and labels. Also if set <br />
the certain parameter("visualize"), can get the visualization of the predictions. <br />
Using model is YoloV4, which finds 8 classes:<br />
          -car
          -truck
          -building
          -boat
          -aircraft
          -vessel
          -railway vehicle
          -engineering vehicle
Model was trained on SkySat and WorldView domain with RGB order of channels. So better predictions will be on the same domains and order.

Before starting of model using you need to install requirements:
```
pip install -r requirements.txt
```
To run inference script need to write the following command:

```
python yolo_inference.py --yolo_path
                         --raster_path
                         --output_path
                         --normalize
                         --bands_order
                         --step
```
Where yolo_path - path to weights of model, 
      raster_path - path to raster,
      output_path - folder where to save output
      normalize - whether normalize raster ( if raster is not 8-bit - True)
      bands_order - order in which to get channels ( R, G, B needed)
      step - window size of crop for model input (default setting = 512)
    
To run inference notebook need to pass PATH_PREPROCESSED_TILE(path to raster) and MODEL_PATH(path to wegihts)

      
      
