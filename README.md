# FGM_Utrecht
##Docker repository of the project: 
[Link to Docker Hub of the project!](https://hub.docker.com/r/gizemsogancioglu/metadata_img)

# Description

    .
    ├── src                         # source code folder. 
    │   ├── metadata_regressor.py   # Main module, training Random Forest regressor with metadata features.  
    │   ├── udiva.py                # Reading the dataset and doing preprocessing.
    ├── data                        # contains metadata information (gender, age, education) and big-5 personality scores of participants for training, validation and test set.    
    ├── features                    # contains feature files.
    └── predictions                 # contains predictions (big-5 personality scores of the test set) by metadata regressor.

# Running the project 
`` Note: Prediction results, which are saved in predictions folder, are mounted to /tmp folder locally.
 Please replace it with any other location that you want to keep output file. ``

```javascript
sudo docker run  -v /tmp:/predictions -it gizemsogancioglu/metadata_img:latest
```
