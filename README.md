# FGM_Utrecht
##Docker repository of the project: 
[Link to Docker Hub of the project!](https://hub.docker.com/r/gizemsogancioglu/metadata_img)

# Description

    .
    ├── src                         # source code folder. 
    │   ├── metadata_regressor.py   # Main module, training Random Forest regressor with metadata features.  
    │   ├── udiva.py  # Reading the dataset and doing preprocessing.
    │   
    └── predictions                 # Matlab project for the arousal prediction [0-2] of the elderly speech data. Please see arousal/README.md for the details. 

# Running the project 
`` Note: Prediction results, which are saved in predictions folder, are mounted to /tmp folder locally.
 Please replace it with any other location that you want to keep output file. ``

```javascript
sudo docker run  -v /tmp:/predictions -it gizemsogancioglu/metadata_img:latest
```
