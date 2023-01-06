# SIP BASIC OBJECT DETECTION

## Download weights
To operate properly component requires weights. Before building image download it from link
`https://drive.google.com/file/d/1icxJEh4Ocs2cO64TKP3DVKheFPtOVmCm/view?usp=sharing`
and place it in folder next to Dockerfile. 
Downloaded file must have the following filename: `best.pt`

## Build image
`docker build -t registry.quantumobile.co/sip_basic_objects_detection .`

## Pull image
`docker pull registry.quantumobile.co/sip_basic_objects_detection`

## Push to registry
`docker push registry.quantumobile.co/sip_basic_objects_detection`

## Docker run command

```
docker run \
    -e "PLANET_ORDER_ID=#############################" \
    -e "OUTPUT_FOLDER=/output" \
    -e "PLANET_API_KEY=###############################" \
    -v `pwd`/data/results:/output \
    registry.quantumobile.co/sip_basic_objects_detections
```

## How to add model to SIP
____

1. Open Admin page, `localhost:9000/admin/`
2. In AOI block select `Components` and click on `+Add`
    * Add <b>Component name</b>: `Add your name`
    * Add <b>Image</b>: `registry.quantumobile.co/sip_basic_objects_detection`
    * Add <b>Additional parameter</b> `PLANET_ORDER_ID`
    * Select <b>Planet API key is required</b>
    * <b>GPU is needed for a component to run</b> could be selected (optional)
3. <b>SAVE</b>
4. Update page with `SIP app` <i>(localhost:3000)</i>
5. Select `Area` or `Field` on the map and save it
6. Drop-down menu on your `Area` or `Field` -> `View reports`
7. `Create new`
8. In `Select layers` choose your component, add additional params like <i>Year</i>, <i>Date range</i> and so on
9. `Save changes`    
