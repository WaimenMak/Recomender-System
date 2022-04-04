# Recomender-System



# How to add heroku: 
1. heroku git:remote -a comp7240-group-project-app
https://comp7240-group-project-app.herokuapp.com/ 


# Docker Client
1. Build client Dockerfile: 
cd client 
docker build . -t recommendation_project/client

2. Start container 
docker run -p 3000:3000 -d recommendation_project/client
     
3. Enter container for debug reasons: 
docker exec -it <container id> /bin/bash




# Docker Backend
1. Build server Dockerfile: 
docker build . -t recommendation_project/server


2. Start conatiner
docker run -p 8000:8000 -d recommendation_project/server

3. Enter container for debug reasons: 
docker exec -it <container id> /bin/bash