# AAVAIL revenue predictor service

## installation 
### install in docker
There is a dockerfile on the root of the project to create the docker image.
Be sure to have docker running on your machine.
Then execute from the root directory:
```
docker build -t <image-name> .
```
this will create an image with name \<image-name> in your doxcker registry.

To run the image type:
```
docker run -d -p 5000:5000 -name <aname>
```
\<aname> is the name of the container created.
the -name parameter is optional if you don't specify docker will create a name for the container.

to verify that the container is running:
```
docker ps
```
this should show something like:
```
```