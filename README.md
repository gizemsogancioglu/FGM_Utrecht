# FGM_Utrecht

##Docker repository of the project: 
[link to Docker Hub of the project!](https://hub.docker.com/r/gizemsogancioglu/metadata_img)

### Prediction results, which are saved in predictions folder, are mounted to /tmp folder locally.
### Please replace it with any other location that you want to keep output file. 
sudo docker run  -v /tmp:/predictions -it gizemsogancioglu/metadata_img:v1
