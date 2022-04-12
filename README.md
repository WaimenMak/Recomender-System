# Recomender-System

This repository has been implemented during the project of COMP 7240. 

If you want to start the recommendation system there are two possible ways: 


## 1. Start via terminal 
#### Start the frontend
1. Navigate to /client -> cd client 
2. npm install // this installs the depencencies for the frontend 
3. node app.js  // this launches the frontend express server 

-> when these steps are successful you can see the index.html file under localhost:3000/ 


#### Start the backend 
1. Navigate to root directory of the project 
2. pip install -r requirements.txt   // to install the requirements 
3. uvicorn main:app                  // to start the 



## 2. Start via docker 
1. docker-compose up --build -d   //This will generate the images necessary for the dockerfiles -> download can take a little 
2. docker-compose up   // To start the containers without build 
3. docker-compose down //To stop the containers 



